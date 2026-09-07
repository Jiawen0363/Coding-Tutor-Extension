#!/usr/bin/env bash
# 串起流程；用 TEST_PIPELINE_PHASE 切阶段（Slurm 两作业共用本脚本，避免多文件）：
#   all（默认）   dialogue →（可选关 vLLM）→ code_gen → coding_test
#   dialogue      仅 run_base + 可选关 vLLM
#   post          仅 code_gen + coding_test
# 用法: bash scripts/run/test.sh <tutor_model> <iteration>
# dialogue 后关 vLLM：STOP_VLLM_AFTER_RUN_BASE=1；Slurm 下配合 VLLM_TUTOR_JOB_PID / VLLM_STUDENT_JOB_PID

set -euo pipefail

if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
  echo "用法: bash $0 <tutor_model> <iteration>" >&2
  echo "可选: TEST_PIPELINE_PHASE=all|dialogue|post（默认 all）" >&2
  exit 1
fi

tutor_model="$1"
iteration="$2"
phase="${TEST_PIPELINE_PHASE:-all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

_vllm_kill_tree() {
  local pid="$1"
  [ -z "$pid" ] || [ ! -e "/proc/$pid" ] && return 0
  local c
  for c in $(pgrep -P "$pid" 2>/dev/null || true); do
    _vllm_kill_tree "$c"
  done
  kill -TERM "$pid" 2>/dev/null || true
}

_vllm_kill_tree_force() {
  local pid="$1"
  [ -z "$pid" ] || [ ! -e "/proc/$pid" ] && return 0
  local c
  for c in $(pgrep -P "$pid" 2>/dev/null || true); do
    _vllm_kill_tree_force "$c"
  done
  kill -KILL "$pid" 2>/dev/null || true
}

_vllm_cleanup_after_dialogue() {
  if [ "${STOP_VLLM_AFTER_RUN_BASE:-0}" != "1" ]; then
    return 0
  fi
  echo "========== 停止 vLLM（STOP_VLLM_AFTER_RUN_BASE=1） =========="
  local _sp="${STUDENT_VLLM_PORT:-8002}"
  local _ports=()
  if [ -n "${VLLM_TUTOR_JOB_PID:-}" ]; then
    _ports+=("${TUTOR_VLLM_PORT:-8001}")
  fi
  _ports+=("$_sp")
  for _port in "${_ports[@]}"; do
    fuser -k "${_port}/tcp" 2>/dev/null || true
  done
  sleep 2
  for _root in "${VLLM_TUTOR_JOB_PID:-}" "${VLLM_STUDENT_JOB_PID:-}"; do
    _vllm_kill_tree "$_root"
  done
  sleep 3
  for _root in "${VLLM_TUTOR_JOB_PID:-}" "${VLLM_STUDENT_JOB_PID:-}"; do
    _vllm_kill_tree_force "$_root"
  done
  for _port in "${_ports[@]}"; do
    fuser -k "${_port}/tcp" 2>/dev/null || true
  done
  sleep 5
}

echo "使用 tutor_model=$tutor_model, iteration=$iteration, TEST_PIPELINE_PHASE=$phase"

if [ "$phase" = "post" ]; then
  echo "========== 2. Code generation (run_code_gen_new.sh) =========="
  bash "$PROJECT_ROOT/scripts/run/run_code_gen_new.sh" "$tutor_model" "$iteration"
  echo "========== 3. Coding test (run_coding_test_final_round.sh) =========="
  bash "$PROJECT_ROOT/scripts/run/run_coding_test_final_round.sh" "$tutor_model" "$iteration"
  echo "========== post 完成 =========="
  exit 0
fi

echo "========== 1. dialogue (run_base.sh) =========="
bash "$PROJECT_ROOT/scripts/run/run_base.sh" "$tutor_model" "$iteration"
_vllm_cleanup_after_dialogue

if [ "$phase" = "dialogue" ]; then
  echo "========== dialogue 完成 =========="
  exit 0
fi

if [ "$phase" != "all" ]; then
  echo "错误: TEST_PIPELINE_PHASE 必须是 all、dialogue 或 post，当前: $phase" >&2
  exit 1
fi

echo "========== 2. Code generation (run_code_gen_new.sh) =========="
bash "$PROJECT_ROOT/scripts/run/run_code_gen_new.sh" "$tutor_model" "$iteration"
echo "========== 3. Coding test (run_coding_test_final_round.sh) =========="
bash "$PROJECT_ROOT/scripts/run/run_coding_test_final_round.sh" "$tutor_model" "$iteration"
echo "========== 全部完成 =========="
