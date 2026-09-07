#!/bin/bash

export PYTHONPATH="/home/wangjian/Coding-Tutor-Extension/traver:$PYTHONPATH"

echo "等待 8001 tutor server 启动中..."
until curl -s -H "Authorization: Bearer EMPTY" http://localhost:8001/v1/models > /dev/null; do
  sleep 5
  echo "等待中..."
done

echo "8001 已准备好，等待 8002 student server 启动中..."
until curl -s -H "Authorization: Bearer EMPTY" http://localhost:8002/v1/models > /dev/null; do
  sleep 5
  echo "等待中..."
done

echo "8002 已准备好，等待 8003 level monitor server 启动中..."
until curl -s -H "Authorization: Bearer EMPTY" http://localhost:8003/v1/models > /dev/null; do
  sleep 5
  echo "等待中..."
done

echo "所有服务已准备好，继续执行。"

cd /home/wangjian/Coding-Tutor-Extension

# Configuration
tutor_model="tutor_adapter"  # LoRA adapter identifier
tutor_setting="vanilla"
iteration="debug"  # Save to debug folder
prompt_element_file=prompt/prompt_elements_final.jsonl  # Use full dataset

# Output directory (get_output_path will add tutor_setting/model_name/student_setting)
output_dir=output/dialogue/moe_ppo_epoch0
model_base_path=/data/models

# Model paths
tutor_model_name_or_path=$tutor_model
student_model_name_or_path=$model_base_path/Mixtral-8x7B-Instruct-v0.1-AWQ
level_monitor_model=$model_base_path/Qwen3-4B

# VLLM endpoints
tutor_vllm_endpoint="http://localhost:8001/v1"
student_vllm_endpoint="http://localhost:8002/v1"
level_monitor_endpoint="http://localhost:8003/v1"

# Student levels to test
student_levels=(low_level med_level high_level)

for level in ${student_levels[@]}; do
    echo "Running interactive tutoring with level monitoring for $level student ..."
    python /home/wangjian/Coding-Tutor-Extension/traver/run_moe.py \
        --tutor_setting $tutor_setting \
        --prompt_element_file $prompt_element_file \
        --output_dir $output_dir \
        --tutor_model_name_or_path $tutor_model_name_or_path \
        --student_model_name_or_path $student_model_name_or_path \
        --student_setting $level \
        --level_monitor_endpoint $level_monitor_endpoint \
        --level_monitor_model $level_monitor_model \
        --vllm_endpoint_tutor $tutor_vllm_endpoint \
        --vllm_endpoint_student $student_vllm_endpoint \
        --show_description false \
        --show_message true
done

echo "✅ All simulations completed!"