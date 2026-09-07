#!/usr/bin/env bash
#SBATCH --job-name=test-pipeline
#SBATCH --output=/data/%u/slurm-logs/%x-%j.out
#SBATCH --error=/data/%u/slurm-logs/%x-%j.err

# 默认 2 卡（dialogue / 单作业全流程）。第二阶段提交时可命令行覆盖，例如：
#   sbatch --gres=gpu:1 --export=ALL,TEST_PIPELINE_PHASE=post ...
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=8:00:00
#
# TEST_PIPELINE_PHASE（默认 all）：
#   all       — 单作业跑完全部（起 vLLM → test.sh）
#   dialogue  — 仅 dialogue，需双卡 + 下面全套 tutor/student 变量
#   post      — 仅 code_gen + coding_test，只需 TUTOR_MODEL ITERATION PROJECT_ROOT（建议另申请 GPU）
#
# 两作业串联：bash scripts/run/submit_test.sh
#
# dialogue / all 必填（无默认值）：
#   TUTOR_MODEL ITERATION PROJECT_ROOT BASE_MODEL_PATH LORA_ADAPTER
#   TUTOR_VLLM_BACKEND TUTOR_VLLM_PORT TUTOR_CUDA STUDENT_VLLM_PORT STUDENT_CUDA STUDENT_MODEL_PATH
# 卡数由 submit 脚本 export PIPELINE_GPUS=1|2 并 sbatch --gres=gpu:N；可选 TUTOR_/STUDENT_GPU_MEMORY_UTILIZATION。
# Qwen 时另需 TUTOR_CHAT_TEMPLATE；remote 时需 TUTOR_VLLM_ENDPOINT TUTOR_VLLM_API_KEY，且只启动本地 student vLLM。

set -euo pipefail

die() { echo "错误: $*" >&2; exit 1; }

_require_set() {
  local n="$1"
  [ -n "${!n:-}" ] || die "环境变量 $n 未设置（本脚本不设默认值，请显式 export）"
}

wait_http_models_ready() {
  local url="$1"
  local name="$2"
  local i
  for i in $(seq 1 180); do
    if curl -sf -H "Authorization: Bearer EMPTY" "${url}/models" >/dev/null; then
      echo "OK: $name ready on ${url}"
      return 0
    fi
    sleep 10
    echo "  waiting $name endpoint ${url} (${i}/180)..."
  done
  echo "Timeout waiting for $name endpoint ${url}"
  return 1
}

PHASE="${TEST_PIPELINE_PHASE:-all}"
case "$PHASE" in
  all|dialogue|post) ;;
  *) die "TEST_PIPELINE_PHASE 必须是 all、dialogue 或 post，当前: $PHASE" ;;
esac

mkdir -p "/data/${USER}/slurm-logs"

echo "Job ${SLURM_JOB_ID:-?} phase=$PHASE on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

source /home/wangjian/miniconda3/etc/profile.d/conda.sh
conda activate jiawen

_cusparse_dir="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cusparselt/lib"
[ -d "$_cusparse_dir" ] && export LD_LIBRARY_PATH="${_cusparse_dir}:${LD_LIBRARY_PATH:-}"
export VLLM_USE_V1=0

if [ "$PHASE" = "post" ]; then
  _require_set TUTOR_MODEL
  _require_set ITERATION
  _require_set PROJECT_ROOT
  cd "$PROJECT_ROOT" || die "无法 cd PROJECT_ROOT=$PROJECT_ROOT"
  export TEST_PIPELINE_PHASE=post
  bash "$PROJECT_ROOT/scripts/run/test.sh" "$TUTOR_MODEL" "$ITERATION"
  echo "Done (post) at $(date)"
  exit 0
fi

_require_set TUTOR_MODEL
_require_set ITERATION
_require_set PROJECT_ROOT

cd "$PROJECT_ROOT" || die "无法 cd PROJECT_ROOT=$PROJECT_ROOT"

USE_REMOTE_VLLM="${USE_REMOTE_VLLM:-0}"
if [ "$USE_REMOTE_VLLM" = "1" ]; then
  _require_set TUTOR_VLLM_ENDPOINT
  _require_set STUDENT_VLLM_ENDPOINT
  wait_http_models_ready "$TUTOR_VLLM_ENDPOINT" "tutor"
  wait_http_models_ready "$STUDENT_VLLM_ENDPOINT" "student"
  export STOP_VLLM_AFTER_RUN_BASE=0
  export TEST_PIPELINE_PHASE="$PHASE"
  bash "$PROJECT_ROOT/scripts/run/test.sh" "$TUTOR_MODEL" "$ITERATION"
  echo "Done (remote vLLM mode) at $(date)"
  exit 0
fi

_require_set TUTOR_VLLM_BACKEND
_require_set STUDENT_VLLM_PORT
_require_set STUDENT_CUDA
_require_set STUDENT_MODEL_PATH

case "${TUTOR_VLLM_BACKEND}" in
  qwen)
    _require_set BASE_MODEL_PATH
    _require_set TUTOR_VLLM_PORT
    _require_set TUTOR_CUDA
    if [ "${TUTOR_USE_BASE:-0}" != "1" ]; then
      _require_set LORA_ADAPTER
      _require_set TUTOR_CHAT_TEMPLATE
    else
      _require_set TUTOR_CHAT_TEMPLATE
    fi
    ;;
  llama)
    _require_set BASE_MODEL_PATH
    _require_set TUTOR_VLLM_PORT
    _require_set TUTOR_CUDA
    if [ "${TUTOR_USE_BASE:-0}" != "1" ]; then
      _require_set LORA_ADAPTER
    fi
    ;;
  remote)
    _require_set TUTOR_VLLM_ENDPOINT
    _require_set TUTOR_VLLM_API_KEY
    ;;
  *) die "TUTOR_VLLM_BACKEND 必须是 qwen、llama 或 remote，当前为: ${TUTOR_VLLM_BACKEND}" ;;
esac

wait_vllm_ready() {
  local port="$1"
  local name="$2"
  local i
  for i in $(seq 1 360); do
    if curl -sf -H "Authorization: Bearer EMPTY" "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "OK: $name ready on port $port"
      return 0
    fi
    sleep 10
    echo "  waiting $name :$port (${i}/360)..."
  done
  echo "Timeout waiting for $name on port $port"
  return 1
}

cleanup_vllm_ports() {
  for _p in "${TUTOR_VLLM_PORT:-}" "$STUDENT_VLLM_PORT"; do
    [ -n "$_p" ] || continue
    fuser -k "${_p}/tcp" 2>/dev/null || true
  done
  sleep 5
}

trap 'cleanup_vllm_ports; exit 130' INT TERM

if [ "$TUTOR_VLLM_BACKEND" = remote ]; then
  echo "========== 使用远程 Tutor API：$TUTOR_VLLM_ENDPOINT（model=$TUTOR_MODEL）=========="
  TUTOR_PID=""
else
  if [ -n "${TUTOR_VLLM_SCRIPT:-}" ]; then
  TUTOR_LAUNCHER="$TUTOR_VLLM_SCRIPT"
  elif [ "${TUTOR_USE_BASE:-0}" = "1" ] && [ "$TUTOR_VLLM_BACKEND" = qwen ]; then
  TUTOR_LAUNCHER="$PROJECT_ROOT/scripts/run/run_qwen_base_tutor.sh"
  elif [ "${TUTOR_USE_BASE:-0}" = "1" ]; then
  TUTOR_LAUNCHER="$PROJECT_ROOT/scripts/run/run_llama_base_tutor.sh"
  elif [ "$TUTOR_VLLM_BACKEND" = qwen ]; then
  TUTOR_LAUNCHER="$PROJECT_ROOT/scripts/run/run_qwen_tutor.sh"
  else
  TUTOR_LAUNCHER="$PROJECT_ROOT/scripts/run/run_llama_tutor.sh"
  fi

  echo "========== 启动 Tutor vLLM：$TUTOR_LAUNCHER（TUTOR_CUDA=$TUTOR_CUDA）=========="
  export TUTOR_VLLM_PORT
  export TUTOR_BASE_MODEL="$BASE_MODEL_PATH"
  if [ "${TUTOR_USE_BASE:-0}" = "1" ]; then
    export TUTOR_SERVED_MODEL_NAME="${TUTOR_VLLM_MODEL_NAME:-$TUTOR_MODEL}"
  else
    export TUTOR_LORA_ADAPTER="$LORA_ADAPTER"
    export TUTOR_LORA_SERVED_NAME="$TUTOR_MODEL"
  fi
  export TUTOR_MODEL_NAME="$TUTOR_MODEL"
  if [ "$TUTOR_VLLM_BACKEND" = qwen ]; then
    export TUTOR_CHAT_TEMPLATE
  fi
  export TUTOR_CUDA

  bash "$TUTOR_LAUNCHER" >>"/data/${USER}/slurm-logs/${SLURM_JOB_ID:-0}-tutor.log" 2>&1 &
  TUTOR_PID=$!
  wait_vllm_ready "$TUTOR_VLLM_PORT" "tutor"
fi

echo "========== 启动 Student vLLM（STUDENT_CUDA=$STUDENT_CUDA）=========="
export STUDENT_CUDA STUDENT_VLLM_PORT STUDENT_MODEL_PATH
bash "$PROJECT_ROOT/scripts/run/run_engine_student.sh" >>"/data/${USER}/slurm-logs/${SLURM_JOB_ID:-0}-student.log" 2>&1 &
STUDENT_PID=$!

wait_vllm_ready "$STUDENT_VLLM_PORT" "student"

export TUTOR_VLLM_PORT="${TUTOR_VLLM_PORT:-}" STUDENT_VLLM_PORT
export STOP_VLLM_AFTER_RUN_BASE=1
export VLLM_TUTOR_JOB_PID="$TUTOR_PID"
export VLLM_STUDENT_JOB_PID="$STUDENT_PID"

if [ "$PHASE" = "dialogue" ]; then
  export TEST_PIPELINE_PHASE=dialogue
  echo "========== 仅 dialogue（TEST_PIPELINE_PHASE=dialogue） =========="
else
  export TEST_PIPELINE_PHASE=all
  echo "========== 全流程（TEST_PIPELINE_PHASE=all） =========="
fi

bash "$PROJECT_ROOT/scripts/run/test.sh" "$TUTOR_MODEL" "$ITERATION"

echo "========== 结束，清理 vLLM 端口 =========="
cleanup_vllm_ports
[ -z "$TUTOR_PID" ] || wait "$TUTOR_PID" 2>/dev/null || true
wait "$STUDENT_PID" 2>/dev/null || true

echo "Done at $(date)"
