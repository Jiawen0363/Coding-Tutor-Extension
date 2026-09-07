#!/usr/bin/env bash
# OhMyGPT gpt-5.5 作 tutor：remote API + 本地 student vLLM（1 卡 dialogue）
# 默认试跑：每个 student level 2 条对话，共 6 条；仅提交 dialogue，不跑 post。
set -euo pipefail

mkdir -p "/data/${USER}/slurm-logs"

ROOT="/data/wangjian/Coding-Tutor-Extension"
API_KEY_FILE="$ROOT/scripts/openai_api_key.txt"

[ -f "$API_KEY_FILE" ] || { echo "错误: 未找到 API key 文件 $API_KEY_FILE" >&2; exit 1; }

export PROJECT_ROOT="$ROOT"
export ITERATION="${ITERATION:-trial_2each}"
export MAX_DIALOGUE_SAMPLES="${MAX_DIALOGUE_SAMPLES:-2}"

export TUTOR_VLLM_BACKEND="remote"
export TUTOR_MODEL="${TUTOR_MODEL:-gpt-5.5}"
export TUTOR_VLLM_ENDPOINT="${TUTOR_VLLM_ENDPOINT:-https://c-z0-api-01.hash070.com/v1}"
export TUTOR_VLLM_API_KEY="$(tr -d '[:space:]' < "$API_KEY_FILE")"

export STUDENT_VLLM_PORT="${STUDENT_VLLM_PORT:-8004}"
export STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/data_old/models/Mixtral-8x7B-Instruct-v0.1-AWQ}"

# shellcheck source=scripts/run/configure_dialogue_gpus.sh
source "$ROOT/scripts/run/configure_dialogue_gpus.sh"

echo "OhMyGPT trial: tutor=$TUTOR_MODEL iteration=$ITERATION max_per_level=$MAX_DIALOGUE_SAMPLES (total=$((MAX_DIALOGUE_SAMPLES * 3)) dialogues)"

RUN_POST_JOB="${RUN_POST_JOB:-0}"
if [ "${RUN_DIALOGUE_JOB:-1}" = "1" ]; then
  JOB1=$(sbatch --parsable --gres="$DIALOGUE_SBATCH_GRES" --export=ALL --gpu-bind="$SLURM_GPU_BIND" "$ROOT/scripts/run/sbatch_job_dialogue.sh")
  echo "Dialogue job: $JOB1 (${DIALOGUE_SBATCH_GRES})"
  if [ "$RUN_POST_JOB" = "1" ]; then
    JOB2=$(sbatch --parsable --export=ALL --dependency=afterok:"$JOB1" "$ROOT/scripts/run/sbatch_job_post.sh")
    echo "Post job (after ${JOB1}): $JOB2"
  else
    echo "跳过 post 阶段（试跑仅 dialogue；全量时 export RUN_POST_JOB=1）"
  fi
else
  JOB2=$(sbatch --parsable --export=ALL "$ROOT/scripts/run/sbatch_job_post.sh")
  echo "Post job: $JOB2"
fi
