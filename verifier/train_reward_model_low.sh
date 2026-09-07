#!/bin/bash

# 多GPU串行训练脚本 - 使用三张卡分别跑三个level，避免资源竞争

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jiawen

# 设置LD_LIBRARY_PATH以包含CUDA库路径（解决libcusparseLt.so.0找不到的问题）
if [ -n "$CONDA_PREFIX" ]; then
    CUSPARSELT_LIB="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparselt/lib"
    CUBLAS_LIB="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib"
    if [ -d "$CUSPARSELT_LIB" ]; then
        export LD_LIBRARY_PATH=$CUSPARSELT_LIB:$CUBLAS_LIB:$LD_LIBRARY_PATH
        echo "已设置LD_LIBRARY_PATH，包含CUDA库路径: $CUSPARSELT_LIB"
    fi
fi

model_base_path=/data/models  # TODO: change to your own path
data_dir=output/reward_model_data
pretrained_model_name_or_path=$model_base_path/Qwen3-8B
eval_parts=(part0 part1 part2 part3 part4)
levels=(low_level)
gpu_ids=(6)  # 三张GPU卡

# 串行执行所有训练任务，避免GPU资源竞争
for i in "${!levels[@]}"; do
    level=${levels[$i]}
    gpu_id=${gpu_ids[$i]}
    
    echo "启动 $level 训练，使用GPU $gpu_id"
    
    # 为每个level的所有part串行训练（避免内存不足）
    for part in ${eval_parts[@]}; do
        echo "Running training on $level of part-$part using GPU $gpu_id ..."
        output_dir=output/reward_model/$level/$part
        
        # 使用CUDA_VISIBLE_DEVICES指定GPU，不使用DeepSpeed，禁用wandb
        CUDA_VISIBLE_DEVICES=$gpu_id WANDB_MODE=disabled python traver/train_reward_model.py \
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
        
        echo "完成 $level/$part 训练"
        
        # 添加短暂延迟，确保GPU资源完全释放
        sleep 5
    done
    
    echo "完成 $level 所有part的训练"
done

echo "所有训练任务完成！"