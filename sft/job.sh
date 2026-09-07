#!/bin/bash
#SBATCH --job-name=qwen3-sft
#SBATCH --output=/data/%u/slurm-logs/%j.out
#SBATCH --error=/data/%u/slurm-logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=8:00:00

# ===== 打印环境信息 =====
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

nvidia-smi

# ===== 激活环境 =====
source /home/wangjian/miniconda3/etc/profile.d/conda.sh
conda activate jiawen

# Disable wandb (cluster may not have WANDB key / no network).
export WANDB_DISABLED=true

# ===== Fix torch CUDA sparse dependency (cusparselt) =====
_cusparse_dirs=()
if [ -n "${CONDA_PREFIX:-}" ]; then
  _cusparse_dirs+=("${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cusparselt/lib")
fi
_cusparse_dirs+=("/home/wangjian/miniconda3/envs/jiawen/lib/python3.10/site-packages/nvidia/cusparselt/lib")
_cusparse_dirs+=("/home/wangjian/miniconda3/lib/python3.10/site-packages/nvidia/cusparselt/lib")
for _d in "${_cusparse_dirs[@]}"; do
  if [ -d "$_d" ]; then export LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"; break; fi
done

# ===== PyTorch 显存优化 =====
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== 进入项目目录 =====
PROJECT_ROOT="/data/wangjian/Coding-Tutor-Extension"
cd "$PROJECT_ROOT/fastchat"

# ===== 训练 =====
tutor_model="Qwen3-8B"
DATA_PATH="$PROJECT_ROOT/validate_reward/output/dialogue/vanilla/first_iter/deepseek-v4-flash/all_levels_combined_train_imitation_deepseek-v4-flash.json"

python finetune.py \
    --model_name_or_path /data/wangjian/models/$tutor_model \
    --data_path "$DATA_PATH" \
    --output_dir "$PROJECT_ROOT/checkpoints/sft-${tutor_model}-deepseek-v4-flash" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type linear \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 4 \
    --fp16 True \
    --gradient_checkpointing True \
    --cutoff_len 2048 \
    --template_name alpaca \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj v_proj \
    --cache_path "$PROJECT_ROOT/caches"

echo "Job finished at: $(date)"
