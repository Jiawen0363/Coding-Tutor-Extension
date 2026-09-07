#!/usr/bin/env bash
# 由 submit_test*.sh source；根据 TUTOR_VLLM_BACKEND 自动设置 dialogue 阶段 GPU 绑定。
#
# 规则：
#   TUTOR_VLLM_BACKEND=remote     API tutor，只起本地 student vLLM，申请 1 张卡
#   TUTOR_VLLM_BACKEND=qwen/llama 本地 tutor + student vLLM，申请 2 张卡
#
# 输出/导出：
#   PIPELINE_GPUS, DIALOGUE_SBATCH_GRES, SLURM_GPU_BIND, TUTOR_CUDA, STUDENT_CUDA（按需）
#   TUTOR_GPU_MEMORY_UTILIZATION, STUDENT_GPU_MEMORY_UTILIZATION（供 run_*_tutor / run_engine_student）

set -euo pipefail

BACKEND="${TUTOR_VLLM_BACKEND:-}"
case "$BACKEND" in
  remote)
    export PIPELINE_GPUS=1
    export DIALOGUE_SBATCH_GRES="gpu:1"
    export STUDENT_CUDA="${STUDENT_CUDA:-0}"
    export SLURM_GPU_BIND="${SLURM_GPU_BIND:-map_gpu:0}"
    export STUDENT_GPU_MEMORY_UTILIZATION="${STUDENT_GPU_MEMORY_UTILIZATION:-0.95}"
    ;;
  qwen|llama)
    export PIPELINE_GPUS=2
    # Allow override so we can request extra GPUs and bind to free physical ones
    # on shared nodes where Slurm accounting ignores non-Slurm occupancy.
    export DIALOGUE_SBATCH_GRES="${DIALOGUE_SBATCH_GRES:-gpu:2}"
    export TUTOR_CUDA="${TUTOR_CUDA:-0}"
    export STUDENT_CUDA="${STUDENT_CUDA:-1}"
    export SLURM_GPU_BIND="${SLURM_GPU_BIND:-map_gpu:0,1}"
    export TUTOR_GPU_MEMORY_UTILIZATION="${TUTOR_GPU_MEMORY_UTILIZATION:-0.70}"
    export STUDENT_GPU_MEMORY_UTILIZATION="${STUDENT_GPU_MEMORY_UTILIZATION:-0.95}"
    ;;
  *)
    echo "错误: TUTOR_VLLM_BACKEND 必须是 remote、qwen 或 llama，当前: ${BACKEND:-未设置}" >&2
    exit 1
    ;;
esac

echo "Dialogue GPU 配置: PIPELINE_GPUS=${PIPELINE_GPUS} backend=${BACKEND} gres=${DIALOGUE_SBATCH_GRES} bind=${SLURM_GPU_BIND:-?} tutor_cuda=${TUTOR_CUDA:-n/a} student_cuda=${STUDENT_CUDA}"
