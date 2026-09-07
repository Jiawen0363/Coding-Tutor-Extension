#!/bin/bash

# 激活 jiawen 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jiawen

# vLLM/PyTorch 需要 libcusparseLt；避免 V1 引擎 triton_key 报错
_cusparse_dir="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cusparselt/lib"
[ -d "$_cusparse_dir" ] && export LD_LIBRARY_PATH="${_cusparse_dir}:${LD_LIBRARY_PATH}"
export VLLM_USE_V1=0

# export CUDA_VISIBLE_DEVICES=0

# #原来的并行配置（使用3张卡）
# gpu_ids=(4 5 6)

# # 新的串行配置（使用1张卡，自动选择显存占用最少的GPU）

# 函数：获取显存占用最少的GPU ID
get_least_used_gpu() {
    # 获取所有GPU的显存信息
    local gpu_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
    
    if [ -z "$gpu_info" ]; then
        echo "错误: 未检测到GPU" >&2
        echo "0"
        return 1
    fi
    
    # 使用awk处理，输出信息到stderr，返回GPU ID到stdout
    echo "$gpu_info" | awk -F', ' '{
        usage = ($2 / $3) * 100
        gpu_id = $1
        printf "GPU %d 显存使用率: %.2f%%\n", gpu_id, usage > "/dev/stderr"
        
        if (NR==1 || usage < min_usage) {
            min_usage = usage
            best_gpu = gpu_id
        }
    } END {
        if (NR == 0) {
            print "0"
            exit 1
        }
        printf "选择显存占用最少的GPU: %d (使用率: %.2f%%)\n", best_gpu, min_usage > "/dev/stderr"
        print best_gpu
    }'
}

# 自动选择显存占用最少的GPU
gpu_id=$(get_least_used_gpu)
if [ -z "$gpu_id" ]; then
    echo "警告: 无法自动选择GPU，使用默认GPU 7"
    gpu_id=7
fi

student_levels=(low_level med_level high_level)
tutor_setting="${TUTOR_SETTING:-vanilla}"
tutor_num_responses=5

# 从命令行参数获取 tutor_model（必需）
if [ -z "$1" ]; then
    echo "错误: 请提供 tutor_model 参数"
    echo "用法: bash $0 <tutor_model>"
    echo "示例: bash $0 Qwen3-8B"
    exit 1
fi

tutor_model="$1"
echo "使用 tutor_model: $tutor_model"

if [ -z "$2" ]; then
    echo "错误: 请提供 iteration 参数"
    echo "用法: bash $0 <tutor_model> <iteration>"
    echo "示例: bash $0 Qwen3-8B-sft-200-ppo-mixed first_iter"
    exit 1
fi

iteration="$2"
echo "使用 iteration: $iteration"

namespace_file=prompt/namespaces_split.json
prompt_element_file=prompt/prompt_elements_final.jsonl

# 本脚本当前用 LM_inference.py 进程内 vLLM，无独立 HTTP 端口；以下仅供注释掉的 GPT 路径参考。
api_key_file=prompt/openai_api_key_gpt4o.txt
azure_endpoint="https://jian-gpt4o.openai.azure.com/"

if [ "$tutor_model" == "gpt-3.5" ]; then
    tutor_model_name_or_path="gpt-3.5"
    api_key_file=prompt/openai_api_key.txt
    azure_endpoint="https://jian-wang-llm-au.openai.azure.com/"
elif [ "$tutor_model" == "gpt-4o" ]; then
    tutor_model_name_or_path="gpt-4o"
elif [ "$tutor_model" == "Meta-Llama-3.1-8B-Instruct" ]; then
    tutor_model_name_or_path=/code/models/Meta-Llama-3.1-8B-Instruct
else
    model_name_or_path=/data/models
fi



student_model_name_or_path=/data_old/models/Mixtral-8x7B-Instruct-v0.1-AWQ
#student_model_name_or_path="gpt-4o"

model_name_or_path=/data_old/models/Mixtral-8x7B-Instruct-v0.1-AWQ
#model_name_or_path="gpt-4o"
prompt_element_file=prompt/prompt_elements_final.jsonl
prompt_base_dir=prompt/student_posttest
DIALOGUE_ROOT="${OUTPUT_DIALOGUE_BASE:-output/dialogue}"
output_base_dir="${OUTPUT_STUDENT_POSTTEST_BASE:-output/student_posttest}"
max_interaction_round=8
max_cognitive_load=60
k="1,3,5,10"
n=10

# Step 1: Create prompts
for level in ${student_levels[@]}; do
    echo "Running making prompt for posttest: tutor_setting: $tutor_setting tutor_model: $tutor_model student_level: $level"
    python traver/utils/make_prompt.py --student_posttest \
        --prompt_element_file $prompt_element_file \
        --simulated_file $DIALOGUE_ROOT/$tutor_setting/$iteration/$tutor_model/$level/simulated_dialogs.jsonl \
        --output_dir $prompt_base_dir/$tutor_setting/$iteration/$tutor_model/$level \
        --student_level $level \
        --max_interaction_round $max_interaction_round \
        --max_cognitive_load $max_cognitive_load
done

# # ====== 原来的并行版本（3张卡同时运行，已注释）======
# # Step 2: Run code generation
# for idx in {0..2}; do
#     level=${student_levels[$idx]}
#     current_gpu=${gpu_ids[$idx]}

#     echo "Running code generation for posttest: tutor_setting: $tutor_setting tutor_model: $tutor_model student_level: $level on GPU: $current_gpu"

#     #CUDA_VISIBLE_DEVICES=$current_gpu python src/gpt_inference.py --model gpt-4o \
#     #    --api_key_file $api_key_file \
#     #    --azure_endpoint $azure_endpoint \
#     #    --prompt_file $prompt_base_dir/$tutor_setting/$tutor_model/$level/prompt_all_rounds.jsonl \
#     #    --output_dir $output_base_dir/$tutor_setting/$tutor_model/$level &  # Run in parallel
#     CUDA_VISIBLE_DEVICES=$current_gpu python traver/utils/LM_inference.py --model $model_name_or_path \
#         --prompt_file $prompt_base_dir/$tutor_setting/$iteration/$tutor_model/$level/prompt_all_rounds.jsonl \
#         --output_dir $output_base_dir/$tutor_setting/$iteration/$tutor_model/$level \
#         --gpu_memory_utilization 0.6 &  # Run in parallel
# done

# Wait for all done
# wait
# # ====== 并行版本结束 ======

# # ====== 新的串行版本（每次运行前自动选择显存最少的GPU）======
# Step 2: Run code generation (串行运行，每次自动选择显存最少的GPU)
for level in ${student_levels[@]}; do
    # 每次运行前重新选择显存占用最少的GPU
    current_gpu_id=$(get_least_used_gpu)
    if [ -z "$current_gpu_id" ]; then
        current_gpu_id=0
    fi
    
    echo "Running code generation for posttest: tutor_setting: $tutor_setting tutor_model: $tutor_model student_level: $level on GPU: $current_gpu_id"

    #CUDA_VISIBLE_DEVICES=$current_gpu_id python src/gpt_inference.py --model gpt-4o \
    #    --api_key_file $api_key_file \
    #    --azure_endpoint $azure_endpoint \
    #    --prompt_file $prompt_base_dir/$tutor_setting/$tutor_model/$level/prompt_all_rounds.jsonl \
    #    --output_dir $output_base_dir/$tutor_setting/$tutor_model/$level
    CUDA_VISIBLE_DEVICES=$current_gpu_id python traver/utils/LM_inference.py --model_name_or_path $model_name_or_path \
        --prompt_file $prompt_base_dir/$tutor_setting/$iteration/$tutor_model/$level/prompt_all_rounds.jsonl \
        --output_dir $output_base_dir/$tutor_setting/$iteration/$tutor_model/$level \
        --context_window 12000 \
        --gpu_memory_utilization 0.6
done
# ====== 串行版本结束 ======

# Step 3: Process completions
for level in ${student_levels[@]}; do
    echo "Processing completions for: tutor_setting: $tutor_setting tutor_model: $tutor_model student_level: $level"
    python traver/utils/process_completion.py \
        --completion_file $output_base_dir/$tutor_setting/$iteration/$tutor_model/$level/completion_lm.jsonl \
        --output_file $output_base_dir/$tutor_setting/$iteration/$tutor_model/$level/completion.jsonl
done
