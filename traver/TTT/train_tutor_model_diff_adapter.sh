#!/bin/bash
conda activate pagoda
BASE_DIR="/code/Coding_Tutor_Extension"
TUTOR_SETTING="vanilla"
TUTOR_MODEL="Qwen3-4B-sft-200"  
ITERATION="first_iter"
TIMESTAMP=$(date +%s)
LOG_DIR="training_logs/${TUTOR_MODEL}_ppo_${TIMESTAMP}"
backbone="Qwen3-4B"

echo "Starting step-ppo training..."
echo "Training $TUTOR_MODEL"
echo "Log directory: $LOG_DIR"

levels=("low_level" "med_level" "high_level")
for level in "${levels[@]}"; do
    echo "=========================="
    echo "Training level: $level"
    echo "=========================="
    
    # 为每个level创建独立的checkpoint目录名
    CHECKPOINT_DIR="checkpoints/${TUTOR_MODEL}_${level}_ppo_${TIMESTAMP}"
    
    # 为每个level设置对应的adapter路径
    if [ "$level" == "low_level" ]; then
        adapter_path="/code/Coding_Tutor_Extension/checkpoints/sft-Qwen3-4B-${level}/checkpoint-200"
    elif [ "$level" == "med_level" ]; then
        adapter_path="/code/Coding_Tutor_Extension/checkpoints/sft-Qwen3-4B-${level}/checkpoint-200"
    elif [ "$level" == "high_level" ]; then
        adapter_path="/code/Coding_Tutor_Extension/checkpoints/sft-Qwen3-4B-${level}/checkpoint-200"
    fi
    
    # 创建level特定的日志目录
    LEVEL_LOG_DIR="${LOG_DIR}/${level}"
    mkdir -p "$LEVEL_LOG_DIR"
    
    echo "Data path: $BASE_DIR/output/ppo_scored/$TUTOR_SETTING/$ITERATION/$TUTOR_MODEL (all levels)"
    echo "Base model: /code/models/$backbone"
    echo "Adapter path: $adapter_path"
    echo "Checkpoint will be saved to: $CHECKPOINT_DIR"
    echo "Logs will be saved to: $LEVEL_LOG_DIR"
    
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python3 traver/TTT/step_ppo.py \
      --config_path traver/TTT/ppo-tutor.json \
      --model_path /code/models/$backbone \
      --adapter_path "$adapter_path" \
      --data_path $BASE_DIR/output/ppo_scored/$TUTOR_SETTING/$ITERATION/$TUTOR_MODEL \
      --model_type TutorModel \
      --tutor_model_name "${TUTOR_MODEL}_${level}" \
      --epochs 3 \
      --log_dir "$LEVEL_LOG_DIR" \
      --checkpoint_dir "${TUTOR_MODEL}_${level}_ppo_${TIMESTAMP}"
    
    echo "Completed training for level: $level"
    echo ""
done

echo "All training completed!"
echo "Training logs saved to: $LOG_DIR"
echo "Checkpoints saved to: checkpoints/"


  # --data_path $BASE_DIR/output/ppo_scored/$TUTOR_SETTING/$ITERATION/$TUTOR_MODEL \

  # --model_path /code/Coding_Tutor_Extension/models/$TUTOR_MODEL \