#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

ROOT=/data/wangjian/EvoCodeBench/EvoCodeBench-2604  # Note: EvoCodeBench-2403 with configured envs
Source_Code_Root=$ROOT/Source_Code
Dependency_Root=$ROOT/Dependency_Data
metadata_file=$ROOT/metadata.jsonl

prompt_element_file=prompt/prompt_elements_final.jsonl
prompt_base_dir=prompt/student_posttest
output_base_dir="${OUTPUT_STUDENT_POSTTEST_BASE:-output/student_posttest}"

# 从命令行参数获取 tutor_model（必需）
if [ -z "$1" ]; then
    echo "错误: 请提供 tutor_model 参数"
    echo "用法: bash $0 <tutor_model>"
    echo "示例: bash $0 Qwen3-4B-sft-200-ppo-epoch0"
    exit 1
fi

tutor_model="$1"
echo "使用 tutor_model: $tutor_model"

tutor_settings=("${TUTOR_SETTING:-vanilla}")
student_levels=(low_level med_level high_level)

k="1,3,5,10"
n=10


# 从命令行参数获取 tutor_model（必需）
if [ -z "$2" ]; then
    echo "错误: 请提供 iteration 参数"
    echo "用法: bash $0 <tutor_model> <iteration>"
    echo "示例: bash $0 Qwen3-4B-sft-200-ppo-epoch0 first_iter"
    exit 1
fi

iteration="$2"
echo "使用 iteration: $iteration"

for setting in ${tutor_settings[@]}; do
    model=$tutor_model
    log_file="$output_base_dir/$setting/$iteration/$model/test_progress.log"
    echo "Starting coding test for $setting/$iteration/$model at $(date)" > $log_file
    
    for level in ${student_levels[@]}; do
        echo "Running recall@$k for tutor_setting: $setting model: $model student_level: $level" | tee -a $log_file
        python traver/parser/recall_k.py \
            --output_file $output_base_dir/$setting/$iteration/$model/$level/completion.jsonl \
            --log_file $output_base_dir/$setting/$iteration/$model/$level/dependency_results.jsonl \
            --data_file $metadata_file \
            --source_code_root $Source_Code_Root \
            --dependency_data_root $Dependency_Root \
            --k $k
        echo "Completed recall@$k for $setting/$iteration/$model/$level at $(date)" | tee -a $log_file
        
        echo "Running pass@$k for tutor_setting: $setting model: $model student_level: $level" | tee -a $log_file
        python traver/utils/check_source_code.py $Source_Code_Root
        python traver/parser/pass_k.py \
            --output_file $output_base_dir/$setting/$iteration/$model/$level/completion.jsonl \
            --log_file $output_base_dir/$setting/$iteration/$model/$level/test_results.jsonl \
            --data_file $metadata_file \
            --source_code_root $Source_Code_Root \
            --k $k \
            --n $n
        echo "Completed pass@$k for $setting/$iteration/$model/$level at $(date)" | tee -a $log_file
    done
    
    echo "All tests completed for $setting/$iteration/$model at $(date)" | tee -a $log_file
done

echo "All tests completed at $(date)"
