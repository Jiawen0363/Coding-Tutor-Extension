#!/usr/bin/env bash
# Student 模拟器 vLLM（无默认值，须显式 export）
#
# 必填：
#   STUDENT_CUDA          — 逻辑 GPU 号（与 tutor 不同卡时常为 1）
#   STUDENT_VLLM_PORT     — 监听端口（须与 run_base 中 student 端口一致）
#   STUDENT_MODEL_PATH    — 学生模型目录（HF 格式，绝对路径）

set -euo pipefail
die() { echo "错误: $*" >&2; exit 1; }
_require() { [ -n "${!1:-}" ] || die "未设置环境变量 $1（run_engine_student.sh 不使用默认值）"; }

_require STUDENT_CUDA
_require STUDENT_VLLM_PORT
_require STUDENT_MODEL_PATH

_cusparse_dir="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cusparselt/lib"
[ -d "$_cusparse_dir" ] && export LD_LIBRARY_PATH="${_cusparse_dir}:${LD_LIBRARY_PATH:-}"
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES="$STUDENT_CUDA"

_student_mem="${STUDENT_GPU_MEMORY_UTILIZATION:-0.95}"
echo "Starting student vLLM: model=$STUDENT_MODEL_PATH port=$STUDENT_VLLM_PORT CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES mem=${_student_mem}"
python -m vllm.entrypoints.openai.api_server \
    --model "$STUDENT_MODEL_PATH" \
    --port "$STUDENT_VLLM_PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$_student_mem" \
    --max-model-len 10380 \
    --enforce-eager \
    --api-key "EMPTY"
