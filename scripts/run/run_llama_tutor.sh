#!/usr/bin/env bash
# Tutor vLLM：Llama 基座 + LoRA（无默认值，须全部 export）
#
# 必填：
#   TUTOR_BASE_MODEL
#   TUTOR_LORA_ADAPTER
#   TUTOR_LORA_SERVED_NAME   — 与 test.sh / run_base 的 TUTOR_MODEL 一致
#   TUTOR_VLLM_PORT
#   TUTOR_CUDA               — 逻辑 GPU 编号（Slurm 单卡作业一般为 0）

set -euo pipefail
die() { echo "错误: $*" >&2; exit 1; }
_require() { [ -n "${!1:-}" ] || die "未设置环境变量 $1（run_llama_tutor.sh 不使用默认值）"; }

_require TUTOR_BASE_MODEL
_require TUTOR_LORA_ADAPTER
_require TUTOR_LORA_SERVED_NAME
_require TUTOR_VLLM_PORT
_require TUTOR_CUDA

_cusparse_dir="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cusparselt/lib"
[ -d "$_cusparse_dir" ] && export LD_LIBRARY_PATH="${_cusparse_dir}:${LD_LIBRARY_PATH:-}"
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES="$TUTOR_CUDA"

_tutor_mem="${TUTOR_GPU_MEMORY_UTILIZATION:-0.6}"
echo "Starting Llama tutor vLLM: base=$TUTOR_BASE_MODEL LoRA=$TUTOR_LORA_ADAPTER served=$TUTOR_LORA_SERVED_NAME port=$TUTOR_VLLM_PORT mem=${_tutor_mem}"
python -m vllm.entrypoints.openai.api_server \
    --model "$TUTOR_BASE_MODEL" \
    --enable-lora \
    --lora-modules "${TUTOR_LORA_SERVED_NAME}=${TUTOR_LORA_ADAPTER}" \
    --max-lora-rank 64 \
    --port "$TUTOR_VLLM_PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$_tutor_mem" \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --enforce-eager \
    --trust-remote-code \
    --api-key "EMPTY"
