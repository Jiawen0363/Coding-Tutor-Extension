#!/bin/bash

export CUDA_VISIBLE_DEVICES="6"
tensor_parallel_size=1
port=8003


model_base_path=/data/models  # TODO: change to your own path
model_name_or_path=$model_base_path/Qwen3-4B
# model_name_or_path=$model_base_path/Llama-3.2-3B-Instruct

echo "Starting vllm engine for $model_name_or_path as tutor agent..."
python -m vllm.entrypoints.openai.api_server \
    --chat-template /home/wangjian/Coding-Tutor-Extension/scripts/run/qwen3_nonthinking.jinja \
    --model $model_name_or_path \
    --port $port \
    --tensor-parallel-size $tensor_parallel_size \
    --gpu-memory-utilization 0.7 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --trust-remote-code \
    --api-key "EMPTY"
