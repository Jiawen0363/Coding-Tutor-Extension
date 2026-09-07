#!/bin/bash

export CUDA_VISIBLE_DEVICES="7"
tensor_parallel_size=1
port=8001


model_base_path=/home/wangjian/Coding-Tutor-Extension
# 基础模型路径 - Qwen3-4B backbone
base_model_path=/data/models/Qwen3-4B

# LoRA adapter 路径 - 三个不同水平的 adapter
low_level_adapter=/home/wangjian/Coding-Tutor-Extension/checkpoints/ppo/Qwen3-4B_MoE_prompt3_low_level_ppo_1760187081/epoch-0
medium_level_adapter=/home/wangjian/Coding-Tutor-Extension/checkpoints/ppo/Qwen3-4B_MoE_prompt3_med_level_ppo_1760187081/epoch-0
high_level_adapter=/home/wangjian/Coding-Tutor-Extension/checkpoints/ppo/Qwen3-4B_MoE_prompt3_high_level_ppo_1760187081/epoch-0

echo "Starting vllm engine with multiple LoRA adapters..."
echo "  - Low level: $low_level_adapter"
echo "  - Medium level: $medium_level_adapter"
echo "  - High level: $high_level_adapter"

python -m vllm.entrypoints.openai.api_server \
    --chat-template /home/wangjian/Coding-Tutor-Extension/scripts/run/qwen3_nonthinking.jinja \
    --model $base_model_path \
    --enable-lora \
    --lora-modules \
        low_level_tutor=$low_level_adapter \
        medium_level_tutor=$medium_level_adapter \
        high_level_tutor=$high_level_adapter \
    --max-lora-rank 64 \
    --port $port \
    --tensor-parallel-size $tensor_parallel_size \
    --gpu-memory-utilization 0.7 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --trust-remote-code \
    --api-key "EMPTY"
