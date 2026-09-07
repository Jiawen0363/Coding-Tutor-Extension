#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
# 优化显存分配，减少碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT="/data/wangjian/Coding-Tutor-Extension"
# 本机 Llama 等在 /data_old/models/（可覆盖：MODEL_ROOT=/data/models bash ...）
MODEL_ROOT="${MODEL_ROOT:-/data_old/models}"

# 使用所有level的数据进行SFT训练
echo "使用所有level的数据进行SFT训练..."
cd "$PROJECT_ROOT/fastchat"

tutor_model="Qwen3-8B"
DATA_PATH="$PROJECT_ROOT/validate_reward/output/dialogue/vanilla/first_iter/deepseek-v4-flash/all_levels_combined_train_imitation_deepseek-v4-flash.json"

python finetune.py \
    --model_name_or_path "$MODEL_ROOT/$tutor_model" \
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