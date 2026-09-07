#!/bin/bash

# 测试脚本 - 只训练一个模型来验证配置
model_base_path=/code/models  # TODO: change to your own path
data_dir=output/reward_model_data
pretrained_model_name_or_path=$model_base_path/Qwen3-8B

# 测试参数 - 只训练high_level的part0
level=high_level
part=part0
output_dir=output/test_reward_model

echo "=== 测试Reward Model训练 ==="
echo "Level: $level"
echo "Part: $part"
echo "Output: $output_dir"
echo "Model: $pretrained_model_name_or_path"
echo "Data: $data_dir/$level"
echo "================================"

# 检查必要文件是否存在
if [ ! -f "$data_dir/$level/verifier_data_$part.jsonl" ]; then
    echo "错误: 数据文件不存在: $data_dir/$level/verifier_data_$part.jsonl"
    exit 1
fi

if [ ! -d "$pretrained_model_name_or_path" ]; then
    echo "错误: 模型路径不存在: $pretrained_model_name_or_path"
    exit 1
fi

# 不再需要DeepSpeed配置文件检查

echo "所有必要文件检查通过，开始训练..."

# 运行训练 - 不使用DeepSpeed
python traver/train_reward_model.py \
    --data_dir $data_dir/$level \
    --eval_part $part \
    --pretrained_model_name_or_path $pretrained_model_name_or_path \
    --output_dir $output_dir \
    --max_length 2200 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --fp16 false \
    --bf16 false \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --eval_steps 500 \
    --save_steps 100 \
    --gradient_checkpointing true

echo "=== 测试完成 ==="
echo "如果训练成功，模型将保存在: $output_dir"
echo "可以检查以下文件："
echo "  - $output_dir/pytorch_model.bin"
echo "  - $output_dir/config.json"
echo "  - $output_dir/training_args.bin"
