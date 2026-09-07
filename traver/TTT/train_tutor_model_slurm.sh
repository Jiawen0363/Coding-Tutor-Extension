#!/bin/bash
#SBATCH --job-name=tutor-ppo
#SBATCH --output=/data/%u/slurm-logs/%j.out
#SBATCH --error=/data/%u/slurm-logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00

# Slurm 版：与 train_tutor_model.sh 同逻辑，在集群上用 sbatch 提交：
#   mkdir -p /data/$USER/slurm-logs
#   cd /data/wangjian/Coding-Tutor-Extension
#   sbatch traver/TTT/train_tutor_model_slurm.sh

set -euo pipefail

echo "Job started at: $(date)"
echo "Node: $(hostname)  JobID: ${SLURM_JOB_ID:-local}"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-}"

nvidia-smi || true

source /home/wangjian/miniconda3/etc/profile.d/conda.sh
conda activate jiawen

export WANDB_DISABLED=true

# cusparselt（torch import）
_cusparse_dirs=()
if [ -n "${CONDA_PREFIX:-}" ]; then
  _cusparse_dirs+=("${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cusparselt/lib")
fi
_cusparse_dirs+=("/home/wangjian/miniconda3/envs/jiawen/lib/python3.10/site-packages/nvidia/cusparselt/lib")
_cusparse_dirs+=("/home/wangjian/miniconda3/lib/python3.10/site-packages/nvidia/cusparselt/lib")
for _d in "${_cusparse_dirs[@]}"; do
  if [ -d "$_d" ]; then export LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"; break; fi
done

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

PROJECT_ROOT="/data/wangjian/Coding-Tutor-Extension"
cd "$PROJECT_ROOT"

# 与 train_tutor_model.sh 对齐（路径改为本机 /data/wangjian）
BASE_DIR="$PROJECT_ROOT"
TUTOR_SETTING="vanilla"
# 须与 output/ppo_scored/.../ 下目录名一致（当前为 Qwen3-8B-sft-200）
TUTOR_MODEL="Qwen3-8B-sft-200"
ITERATION="first_iter"
TIMESTAMP=$(date +%s)
LOG_DIR="training_logs/${TUTOR_MODEL}_all_levels_${TIMESTAMP}"
backbone="Qwen3-8B"
ADAPTER_PATH="/data/wangjian/Coding-Tutor-Extension/checkpoints/sft-Qwen3-8B_self/checkpoint-200"

mkdir -p "$LOG_DIR"

DATA_DIR="$BASE_DIR/output/ppo_scored/$TUTOR_SETTING/$ITERATION/${TUTOR_MODEL}"

echo "Using data: $DATA_DIR"
echo "Adapter: $ADAPTER_PATH"
echo "Base model: /data/wangjian/models/$backbone"
echo "Log: $LOG_DIR  checkpoint: checkpoints/${TUTOR_MODEL}_ppo_${TIMESTAMP}"

python3 traver/TTT/step_ppo.py \
  --config_path traver/TTT/ppo-tutor.json \
  --adapter_path "$ADAPTER_PATH" \
  --model_path "/data/wangjian/models/$backbone" \
  --data_path "$DATA_DIR" \
  --model_type TutorModel \
  --tutor_model_name "${TUTOR_MODEL}" \
  --epochs 2 \
  --log_dir "$LOG_DIR" \
  --checkpoint_dir "${TUTOR_MODEL}_ppo_${TIMESTAMP}"

echo "Job finished at: $(date)"
echo "Logs: $LOG_DIR"
echo "Checkpoints: ${PROJECT_ROOT}/checkpoints/${TUTOR_MODEL}_ppo_${TIMESTAMP}"
