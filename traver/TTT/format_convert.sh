#!/bin/bash
set -euo pipefail

# 初始化conda
eval "$(conda shell.bash hook)"

# 设置基础目录（与 run_scoring.sh 保持一致）
base_dir="/data/wangjian/Coding-Tutor-Extension"
cd "$base_dir" || exit 1

# 若需 PyTorch/CUDA，确保能找到 libcusparseLt.so.0
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda-11/lib64:${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH}"

# 激活conda环境
conda activate pagoda

# 检查是否提供了MODEL_NAME参数
if [ $# -ne 1 ]; then
    echo "Error: Please provide MODEL_NAME as the first argument"
    echo "Usage: $0 <MODEL_NAME>"
    exit 1
fi

STUDENT_LEVEL=("low_level" "med_level" "high_level")
TUTOR_SETTING="vanilla"
ITERATION="first_iter"
# 设置参数
PROMPT_ELEMENTS_FILE="prompt/prompt_elements_final.jsonl"
MODEL_NAME="$1"  # 用于输入/输出目录命名
# run_scoring.sh 的输出目录是 ${TUTOR_MODEL}_gpt，这里保持一致读取

# tokenizer 仅用于 prompt 构造，固定使用本地路径避免误走 huggingface.co/nil
MODEL_NAME_OR_PATH="/data_old/models/Llama-3.1-8B-Instruct"


for level in "${STUDENT_LEVEL[@]}"; do
    INPUT_FILE="output/scored/$TUTOR_SETTING/$ITERATION/$MODEL_NAME/$level/simulated_dialogs_scoring.json"
    # if [ "$MODEL_NAME" == "Qwen3-4B-sft-279" ] || [ "$MODEL_NAME" == "Qwen3-4B-sft-200" ]; then
    OUTPUT_FILE="output/ppo_scored/$TUTOR_SETTING/$ITERATION/$MODEL_NAME/${MODEL_NAME}_${level}.jsonl"

    # else
    #     echo "Error: Invalid model name"
    #     exit 1
    # fi

# 确保输出目录存在
mkdir -p "$(dirname "$OUTPUT_FILE")"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: input file not found: $INPUT_FILE"
    exit 1
fi

# 运行格式转换
python3 traver/TTT/format_convert.py \
    --prompt_elements_file "$PROMPT_ELEMENTS_FILE" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE"

echo "✅ 格式转换完成: $level"
done