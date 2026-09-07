#!/bin/bash

# 一键获取模型评估结果的脚本（环境修复版）
# 使用方法: bash scripts/eval/get_model_results_env_fixed.sh [model_name]

# 设置环境变量
ROOT=/home/wangjian/EvoCodeBench/EvoCodeBench-2403
Source_Code_Root=$ROOT/Source_Code
Dependency_Root=$ROOT/Dependency_Data
metadata_file=$ROOT/metadata.jsonl
output_base_dir=output/student_posttest
iteration=first_iter
# 默认模型名称
DEFAULT_MODEL="Qwen3-4B-sft-moe-ppo"
MODEL_NAME=${1:-$DEFAULT_MODEL}

# 检查模型目录是否存在
model_dir="$output_base_dir/vanilla/$iteration/$MODEL_NAME"
if [ ! -d "$model_dir" ]; then
    echo "❌ 错误: 模型目录不存在: $model_dir"
    echo "请检查模型名称是否正确，或者先运行测试脚本"
    exit 1
fi

echo "📊 正在获取 $MODEL_NAME 的评估结果..."
echo "=========================================="

# 确保激活正确的conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pagoda

# # 验证环境
# echo "✅ 当前环境: $(conda info --envs | grep '*' | awk '{print $1}')"
# echo "✅ Python路径: $(which python)"

# 函数：获取单个级别的结果
get_level_results() {
    local level=$1
    local model_path="$model_dir/$level"
    
    echo "=== $level ==="
    
    # 检查文件是否存在
    if [ ! -f "$model_path/completion.jsonl" ]; then
        echo "❌ 错误: completion.jsonl 不存在"
        return
    fi
    
    if [ ! -f "$model_path/dependency_results.jsonl" ]; then
        echo "❌ 错误: dependency_results.jsonl 不存在"
        return
    fi
    
    if [ ! -f "$model_path/test_results.jsonl" ]; then
        echo "❌ 错误: test_results.jsonl 不存在"
        return
    fi
    
    # 获取 recall@k 结果
    echo "📈 Recall@k 结果:"
    python traver/parser/recall_k.py \
        --output_file "$model_path/completion.jsonl" \
        --log_file "$model_path/dependency_results.jsonl" \
        --data_file "$metadata_file" \
        --source_code_root "$Source_Code_Root" \
        --dependency_data_root "$Dependency_Root" \
        --k "1,3,5,10" 2>/dev/null | grep "Recall@"
    
    echo ""
    
    # 获取 pass@k 结果
    echo "✅ Pass@k 结果:"
    python traver/parser/pass_k.py \
        --output_file "$model_path/completion.jsonl" \
        --log_file "$model_path/test_results.jsonl" \
        --data_file "$metadata_file" \
        --source_code_root "$Source_Code_Root" \
        --k "1,3,5,10" \
        --n 10 2>/dev/null | grep "Pass@"
    
    echo ""
}

# 获取所有级别的结果
for level in low_level med_level high_level; do
    get_level_results $level
done

echo "=========================================="
echo " $MODEL_NAME 评估完成！"

