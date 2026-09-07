#!/bin/bash
# tutor_model="Llama-3.2-3B-Instruct"
# export PYTHONPATH="/home/wangjian/Coding-Tutor-Extension/traver:$PYTHONPATH"
# echo "等待 8100 tutor server 启动中..."

# until curl -s -H "Authorization: Bearer EMPTY" http://localhost:8100/v1/models > /dev/null; do
#   sleep 5
#   echo "等待中..."
# done

# echo "8100 已准备好，等待 8200 student server 启动中..."

# until curl -s -H "Authorization: Bearer EMPTY" http://localhost:8200/v1/models > /dev/null; do
#   sleep 5
#   echo "等待中..."
# done

# echo "8200 已准备好，继续执行。"


PROJECT_ROOT="/data/wangjian/Coding-Tutor-Extension"
cd "$PROJECT_ROOT"
# 支持：bash run_base.sh <tutor_model> <iteration>；未传则用默认
if [ -n "${1:-}" ]; then tutor_model="$1"; else tutor_model="Llama-3.1-8B-Instruct-sft-200-ppo"; fi
if [ -n "${2:-}" ]; then iteration="$2"; else iteration="first_iter"; fi
tutor_setting="${TUTOR_SETTING:-vanilla}"
tutor_prompt_setting="${TUTOR_PROMPT_SETTING:-base}"
prompt_element_file="${PROMPT_ELEMENT_FILE:-prompt/prompt_elements_final.jsonl}"
DIALOGUE_ROOT="${OUTPUT_DIALOGUE_BASE:-output/dialogue}"
output_dir=$DIALOGUE_ROOT/$tutor_setting/$iteration/$tutor_model
# 与 run_qwen_tutor.sh / run_engine_student.sh 端口一致（可用环境变量覆盖）
TUTOR_VLLM_PORT="${TUTOR_VLLM_PORT:-8001}"
STUDENT_VLLM_PORT="${STUDENT_VLLM_PORT:-8002}"
tutor_vllm_endpoint="${TUTOR_VLLM_ENDPOINT:-http://localhost:${TUTOR_VLLM_PORT}/v1}"
tutor_api_key_file="${TUTOR_API_KEY_FILE:-$PROJECT_ROOT/scripts/openai_api_key.txt}"
if [ -n "${TUTOR_VLLM_API_KEY:-}" ]; then
    tutor_vllm_api_key="$TUTOR_VLLM_API_KEY"
elif [ -f "$tutor_api_key_file" ] && [[ "$tutor_vllm_endpoint" == https://* ]]; then
    tutor_vllm_api_key="$(tr -d '[:space:]' < "$tutor_api_key_file")"
else
    tutor_vllm_api_key="${VLLM_API_KEY:-EMPTY}"
fi
model_base_path=/data/wangjian/models

# tutor_model_name_or_path：请求 vLLM API 的 model 名；可用 TUTOR_VLLM_MODEL_NAME 覆盖
if [ -n "${TUTOR_VLLM_MODEL_NAME:-}" ]; then
    tutor_model_name_or_path="$TUTOR_VLLM_MODEL_NAME"
elif [ "$tutor_model" == "gpt-3.5" ]; then
    tutor_model_name_or_path="gpt-3.5"
elif [ "$tutor_model" == "Qwen3-4B-sft-200-ppo-mixed" ]; then
    tutor_model_name_or_path=$model_base_path/Qwen3-4B-sft-200-ppo-mixed
elif [ "$tutor_model" == "Llama-3.1-8B-Instruct" ]; then
    tutor_model_name_or_path="Llama-3.1-8B-Instruct"
else
    tutor_model_name_or_path=$tutor_model
fi

student_model_name_or_path=/data_old/models/Mixtral-8x7B-Instruct-v0.1-AWQ
student_vllm_endpoint="${STUDENT_VLLM_ENDPOINT:-http://localhost:${STUDENT_VLLM_PORT}/v1}"
student_vllm_api_key="${STUDENT_VLLM_API_KEY:-EMPTY}"
student_levels=(low_level med_level high_level)

for level in ${student_levels[@]}; do
    echo "Running interactive tutoring between tutor and $level student ..."
    _run_args=(
        --tutor_setting "$tutor_setting"
        --tutor_prompt_setting "$tutor_prompt_setting"
        --prompt_element_file "$prompt_element_file"
        --output_dir "$output_dir"
        --tutor_model_name_or_path "$tutor_model_name_or_path"
        --student_model_name_or_path "$student_model_name_or_path"
        --student_setting "$level"
        --vllm_endpoint_tutor "$tutor_vllm_endpoint"
        --vllm_endpoint_student "$student_vllm_endpoint"
        --tutor_vllm_api_key "$tutor_vllm_api_key"
        --student_vllm_api_key "$student_vllm_api_key"
        --show_description false
        --show_message true
    )
    if [ -n "${MAX_DIALOGUE_SAMPLES:-}" ]; then
        _run_args+=(--max_samples "$MAX_DIALOGUE_SAMPLES")
    fi
    python "$PROJECT_ROOT/traver/run_base.py" "${_run_args[@]}"
done

# bash /data_new/wangjian/Coding-Tutor-Extension/scripts/run/run_code_gen_new.sh $tutor_model $iteration
# # sleep 18000   # 可选：间隔 5 小时后再跑下一段时取消注释
# bash /data_new/wangjian/Coding-Tutor-Extension/scripts/run/run_coding_test_final_round.sh $tutor_model $iteration