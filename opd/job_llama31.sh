#!/bin/bash
#SBATCH --job-name=llama31-opd-pi
#SBATCH --output=/data/%u/slurm-logs/%x-%j.out
#SBATCH --error=/data/%u/slurm-logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00

set -euo pipefail

echo "Job started at: $(date)"
echo "Node: $(hostname)  JobID: ${SLURM_JOB_ID:-local}"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-}"

nvidia-smi || true

source /home/wangjian/miniconda3/etc/profile.d/conda.sh
conda activate jiawen

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRL_EXPERIMENTAL_SILENCE="${TRL_EXPERIMENTAL_SILENCE:-1}"

_cusparse_dirs=()
if [ -n "${CONDA_PREFIX:-}" ]; then
  _cusparse_dirs+=("${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cusparselt/lib")
fi
_cusparse_dirs+=("/home/wangjian/miniconda3/envs/jiawen/lib/python3.10/site-packages/nvidia/cusparselt/lib")
_cusparse_dirs+=("/home/wangjian/miniconda3/envs/jiawen/lib/python3.10/site-packages/nvidia/cusparselt/lib")
for _d in "${_cusparse_dirs[@]}"; do
  if [ -d "$_d" ]; then export LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"; break; fi
done

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangjian/Coding-Tutor-Extension}"
mkdir -p "/data/${USER}/slurm-logs"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +%s)

export STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/data_old/models/Llama-3.1-8B-Instruct}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-$PROJECT_ROOT/checkpoints/sft-Llama-3.1-8B-Instruct/checkpoint-200}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/checkpoints/OPD/Llama-3.1-8B-Instruct-$TIMESTAMP}"
export TRAINER="${TRAINER:-sdft}"
export TEACHER_MODEL_KIND="${TEACHER_MODEL_KIND:-base}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_LENGTH="${MAX_LENGTH:-4352}"
unset CHAT_TEMPLATE

echo "Student model: ${STUDENT_MODEL_PATH}"
echo "Teacher checkpoint: ${TEACHER_MODEL_PATH}"
echo "Trainer: ${TRAINER} (teacher=${TEACHER_MODEL_KIND}, same backbone)"
echo "Output dir: ${OUTPUT_DIR}"

bash "$PROJECT_ROOT/opd/run_opd_training.sh"

echo "Job finished at: $(date)"
