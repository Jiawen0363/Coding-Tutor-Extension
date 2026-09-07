#!/usr/bin/env bash
# Tutor vLLM：Llama 基座（无 LoRA）
#
# 必填：
#   TUTOR_BASE_MODEL
#   TUTOR_VLLM_PORT
#   TUTOR_CUDA

set -euo pipefail
die() { echo "错误: $*" >&2; exit 1; }
_require() { [ -n "${!1:-}" ] || die "未设置环境变量 $1（run_llama_base_tutor.sh 不使用默认值）"; }

_require TUTOR_BASE_MODEL
_require TUTOR_VLLM_PORT
_require TUTOR_CUDA

_cusparse_dir="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cusparselt/lib"
[ -d "$_cusparse_dir" ] && export LD_LIBRARY_PATH="${_cusparse_dir}:${LD_LIBRARY_PATH:-}"
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES="$TUTOR_CUDA"

_tutor_mem="${TUTOR_GPU_MEMORY_UTILIZATION:-0.6}"
_served="${TUTOR_SERVED_MODEL_NAME:-${TUTOR_MODEL_NAME:-$TUTOR_BASE_MODEL}}"
echo "Starting Llama base tutor vLLM: base=$TUTOR_BASE_MODEL served=$_served port=$TUTOR_VLLM_PORT mem=${_tutor_mem}"
python -m vllm.entrypoints.openai.api_server \
    --model "$TUTOR_BASE_MODEL" \
    --served-model-name "$_served" \
    --port "$TUTOR_VLLM_PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$_tutor_mem" \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --enforce-eager \
    --trust-remote-code \
    --api-key "EMPTY"
