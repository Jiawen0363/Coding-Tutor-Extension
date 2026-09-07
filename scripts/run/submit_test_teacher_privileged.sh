#!/usr/bin/env bash
# 评测 base teacher（Llama / Qwen）作为 tutor，使用 privileged information（student prompt + z）。
# 输出目录：
#   dialogue/.../vanilla/backbone/Llama-3.1-8B-Instruct-pt/
#   dialogue/.../vanilla/backbone/Qwen3-8B-pt/
set -euo pipefail

ROOT="/data/wangjian/Coding-Tutor-Extension"
export PROJECT_ROOT="$ROOT"
export OUTPUT_DIALOGUE_BASE="$ROOT/validate_reward/output/dialogue"
export OUTPUT_STUDENT_POSTTEST_BASE="$ROOT/validate_reward/output/student_posttest"
export ITERATION="${ITERATION:-backbone}"
export RUN_DIALOGUE_JOB=1

export TUTOR_USE_BASE=1
export TUTOR_SETTING=vanilla
export TUTOR_PROMPT_SETTING=teacher
export TUTOR_VLLM_PORT="${TUTOR_VLLM_PORT:-8001}"
export STUDENT_VLLM_PORT="${STUDENT_VLLM_PORT:-8002}"
export STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/data_old/models/Mixtral-8x7B-Instruct-v0.1-AWQ}"
export TUTOR_GPU_MEMORY_UTILIZATION="${TUTOR_GPU_MEMORY_UTILIZATION:-0.55}"
export STUDENT_GPU_MEMORY_UTILIZATION="${STUDENT_GPU_MEMORY_UTILIZATION:-0.75}"

unset EXTRA_DEP_JOB

echo "========== Teacher backbone eval: Llama =========="
export TUTOR_VLLM_BACKEND=llama
export TUTOR_MODEL="Llama-3.1-8B-Instruct-pt"
export TUTOR_VLLM_MODEL_NAME="Llama-3.1-8B-Instruct"
export BASE_MODEL_PATH="/data_old/models/Llama-3.1-8B-Instruct"
unset LORA_ADAPTER TUTOR_CHAT_TEMPLATE
_llama_out="$(bash "$ROOT/scripts/run/submit_test.sh")"
echo "$_llama_out"
LLAMA_POST="$(echo "$_llama_out" | grep 'Post job' | tail -1 | awk '{print $NF}')"
echo "Llama teacher post job: $LLAMA_POST"

echo "========== Teacher backbone eval: Qwen（after Llama post $LLAMA_POST）=========="
export EXTRA_DEP_JOB="$LLAMA_POST"
export TUTOR_VLLM_BACKEND=qwen
export TUTOR_MODEL="Qwen3-8B-pt"
export TUTOR_VLLM_MODEL_NAME="Qwen3-8B"
export BASE_MODEL_PATH="/data_old/models/Qwen3-8B"
export TUTOR_CHAT_TEMPLATE="$ROOT/scripts/run/qwen3_nonthinking.jinja"
bash "$ROOT/scripts/run/submit_test.sh"
