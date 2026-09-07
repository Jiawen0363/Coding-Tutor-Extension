#!/usr/bin/env bash
# 提交两作业串联：dialogue（API tutor 自动 1 卡）→ post（1 卡）
# 卡数由 TUTOR_VLLM_BACKEND 自动决定：remote/API 用 1 卡；qwen/llama 本地模型用 2 卡。
set -euo pipefail

mkdir -p "/data/${USER}/slurm-logs"

ROOT="/data/wangjian/Coding-Tutor-Extension"

export PROJECT_ROOT="$ROOT"
# # 可选：仅当下面两行**保留**时，dialogue / post 写到 validate_reward；注释掉则走默认
#   output/dialogue 与 output/student_posttest（与 sbatch 是否 --export=ALL 一致）
# export OUTPUT_DIALOGUE_BASE="$ROOT/validate_reward/output/dialogue"
# export OUTPUT_STUDENT_POSTTEST_BASE="$ROOT/validate_reward/output/student_posttest"


export ITERATION="${ITERATION:-first_iter}"

# # DeepSeek V4 tutor：远程 OpenAI-compatible API，不启动本地 tutor vLLM。
# # 仅在 TUTOR_VLLM_BACKEND=remote 时需要 DEEPSEEK_API_KEY。
# if [ "${TUTOR_VLLM_BACKEND:-remote}" = "remote" ]; then
#   : "${DEEPSEEK_API_KEY:?请先 export DEEPSEEK_API_KEY=你的 DeepSeek API key}"
#   export TUTOR_VLLM_BACKEND="remote"
#   export TUTOR_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-pro}"
#   export TUTOR_VLLM_ENDPOINT="https://api.deepseek.com/v1"
#   export TUTOR_VLLM_API_KEY="$DEEPSEEK_API_KEY"
# fi

# 本地 LoRA tutor 配置仅在 TUTOR_VLLM_BACKEND=qwen/llama 时需要。
export BASE_MODEL_PATH="/data_old/models/Qwen3-8B"
export LORA_ADAPTER="/data/wangjian/Coding-Tutor-Extension/checkpoints/OPD/Qwen3-8B-1787576349/checkpoint-135"

# remote 用 8004；本地 llama/qwen 默认 8002（可用环境变量覆盖）
if [ "${TUTOR_VLLM_BACKEND:-remote}" = "remote" ]; then
  export STUDENT_VLLM_PORT="${STUDENT_VLLM_PORT:-8004}"
else
  export STUDENT_VLLM_PORT="${STUDENT_VLLM_PORT:-8002}"
fi
export STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/data_old/models/Mixtral-8x7B-Instruct-v0.1-AWQ}"

# shellcheck source=scripts/run/configure_dialogue_gpus.sh
source "$ROOT/scripts/run/configure_dialogue_gpus.sh"

export TUTOR_CHAT_TEMPLATE="$ROOT/scripts/run/qwen3_nonthinking.jinja"

# 两个不同的提交入口（资源与阶段不同）；共用 sbatch_run_test.sh 里的 bash 逻辑，避免维护两份实现。
# 默认只提交 post。若要串联提交 dialogue + post，先设置 RUN_DIALOGUE_JOB=1。
# 若需等已有作业结束再提交本 pipeline，设置 EXTRA_DEP_JOB（如 export EXTRA_DEP_JOB=1239）。
export RUN_DIALOGUE_JOB="${RUN_DIALOGUE_JOB:-1}"

_dialogue_sbatch_deps() {
  if [ -n "${EXTRA_DEP_JOB:-}" ]; then
    echo "afterok:${EXTRA_DEP_JOB}"
  fi
}

if [ "${RUN_DIALOGUE_JOB:-0}" = "1" ]; then
  _dep="$(_dialogue_sbatch_deps)"
  if [ -n "$_dep" ]; then
    JOB1=$(sbatch --parsable --gres="$DIALOGUE_SBATCH_GRES" --export=ALL --gpu-bind="$SLURM_GPU_BIND" --dependency="$_dep" "$ROOT/scripts/run/sbatch_job_dialogue.sh")
    echo "Dialogue job: $JOB1 (${DIALOGUE_SBATCH_GRES}, after job ${EXTRA_DEP_JOB})"
  else
    JOB1=$(sbatch --parsable --gres="$DIALOGUE_SBATCH_GRES" --export=ALL --gpu-bind="$SLURM_GPU_BIND" "$ROOT/scripts/run/sbatch_job_dialogue.sh")
    echo "Dialogue job: $JOB1 (${DIALOGUE_SBATCH_GRES})"
  fi
  JOB2=$(sbatch --parsable --export=ALL --dependency=afterok:"$JOB1" "$ROOT/scripts/run/sbatch_job_post.sh")
  echo "Post job (after ${JOB1}): $JOB2"
elif [ "${RUN_DIALOGUE_JOB:-0}" = "2" ]; then
  JOB1=$(sbatch --parsable --gres="$DIALOGUE_SBATCH_GRES" --export=ALL --gpu-bind="$SLURM_GPU_BIND" "$ROOT/scripts/run/sbatch_job_dialogue.sh")
  echo "Dialogue job: $JOB1 (${DIALOGUE_SBATCH_GRES})"
elif [ "${RUN_DIALOGUE_JOB:-0}" = "0" ]; then
  _post_dep=""
  if [ -n "${EXTRA_DEP_JOB:-}" ]; then
    _post_dep="--dependency=afterok:${EXTRA_DEP_JOB}"
  fi
  # shellcheck disable=SC2086
  JOB2=$(sbatch --parsable --export=ALL ${_post_dep} "$ROOT/scripts/run/sbatch_job_post.sh")
  echo "Post job: $JOB2"
else
  echo "错误: RUN_DIALOGUE_JOB 必须是 0、1 或 2，当前: ${RUN_DIALOGUE_JOB:-}" >&2
  exit 1
fi
