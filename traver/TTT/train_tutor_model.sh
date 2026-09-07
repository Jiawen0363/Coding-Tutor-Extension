#!/bin/bash
# 本地直接跑：bash traver/TTT/train_tutor_model.sh（需在仓库根目录）
# 集群 Slurm：sbatch traver/TTT/train_tutor_model_slurm.sh

# 指定使用的 GPU（单卡；Slurm 训练请用 train_tutor_model_slurm.sh）
export CUDA_VISIBLE_DEVICES=0

BASE_DIR="/data/wangjian/Coding-Tutor-Extension"
TUTOR_SETTING="vanilla"
# 须与 output/ppo_scored/.../ 下目录名一致（当前为 Qwen3-8B-sft-200）
TUTOR_MODEL="Llama-3.1-8B-Instruct-sft-200_self"
ITERATION="first_iter"
TIMESTAMP=$(date +%s)
LOG_DIR="training_logs/${TUTOR_MODEL}_all_levels_${TIMESTAMP}"
backbone="Llama-3.1-8B-Instruct"
ADAPTER_PATH="/data/wangjian/Coding-Tutor-Extension/checkpoints/sft-Llama-3.1-8B-Instruct_self/checkpoint-200"
echo "Starting step-ppo training..."
echo "Training $TUTOR_MODEL with all levels data"
echo "Log directory: $LOG_DIR"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 使用所有level的数据目录
DATA_DIR="$BASE_DIR/output/ppo_scored/$TUTOR_SETTING/$ITERATION/Llama-3.1-8B-Instruct-sft-200"

echo "Using data from all levels in directory: $DATA_DIR"
echo "Loading adapter from: $ADAPTER_PATH"
echo "Data files:"
echo "  - ${TUTOR_MODEL}_low_level.jsonl"
echo "  - ${TUTOR_MODEL}_med_level.jsonl" 
echo "  - ${TUTOR_MODEL}_high_level.jsonl"
echo "Data path: $DATA_DIR"
echo "Base model: /data/wangjian/models/$backbone"
echo "Checkpoint will be saved to: checkpoints/${TUTOR_MODEL}_ppo_${TIMESTAMP}"
echo "Logs will be saved to: $LOG_DIR"

# 添加 cusparseLt 库路径到 LD_LIBRARY_PATH（解决 libcusparseLt.so.0 找不到的问题）
export LD_LIBRARY_PATH=/home/wangjian/miniconda3/envs/jiawen/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH

# 设置内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

python3 traver/TTT/step_ppo.py \
  --config_path traver/TTT/ppo-tutor.json \
  --adapter_path "$ADAPTER_PATH" \
  --model_path "/data_old/models/$backbone" \
  --data_path "$DATA_DIR" \
  --model_type TutorModel \
  --tutor_model_name "${TUTOR_MODEL}" \
  --epochs 2 \
  --log_dir "$LOG_DIR" \
  --checkpoint_dir "${TUTOR_MODEL}_ppo_self_${TIMESTAMP}"
    
echo "All training completed!"
echo "Training logs saved to: $LOG_DIR"
echo "Checkpoints saved to: checkpoints/${TUTOR_MODEL}_ppo_self_${TIMESTAMP}"


  # --data_path $BASE_DIR/output/ppo_scored/$TUTOR_SETTING/$ITERATION/$TUTOR_MODEL \

  # --model_path /code/Coding_Tutor_Extension/models/$TUTOR_MODEL \