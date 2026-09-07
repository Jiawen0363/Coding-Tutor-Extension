# !/bin/bash
export CUDA_VISIBLE_DEVICES=1

# Fix torch CUDA sparse dependency (cusparselt).
_cusparse_dirs=()
if [ -n "${CONDA_PREFIX:-}" ]; then
  _cusparse_dirs+=("${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cusparselt/lib")
fi
_cusparse_dirs+=("/home/wangjian/miniconda3/envs/jiawen/lib/python3.10/site-packages/nvidia/cusparselt/lib")
_cusparse_dirs+=("/home/wangjian/miniconda3/lib/python3.10/site-packages/nvidia/cusparselt/lib")
for _d in "${_cusparse_dirs[@]}"; do
  if [ -d "$_d" ]; then export LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"; break; fi
done

tutor_setting="traver"
tutor_model="gpt-4o"

namespace_file=prompt/namespaces.json
prompt_element_file=prompt/prompt_elements_final.jsonl
output_dir=output/dialogue

tutor_vllm_endpoint="http://localhost:8001/v1"
api_key_file="/data/wangjian/Coding-Tutor-Extension/scripts/openai_api_key.txt"  # TODO: change to your own Azure API key
azure_endpoint="https://api.chatanywhere.tech/v1"  # TODO: change to your own Azure endpoint
model_base_path=/data/wangjian/models  # TODO: change to your own path

if [ "$tutor_model" == "gpt-3.5" ]; then
    tutor_model_name_or_path="gpt-3.5"
elif [ "$tutor_model" == "gpt-4o" ]; then
    tutor_model_name_or_path="gpt-4o"
elif [ "$tutor_model" == "Meta-Llama-3.1-8B-Instruct" ]; then
    tutor_model_name_or_path=$model_base_path/Meta-Llama-3.1-8B-Instruct
elif [ "$tutor_model" == "Meta-Llama-3.1-70B-Instruct" ]; then
    tutor_model_name_or_path=$model_base_path/Meta-Llama-3.1-70B-Instruct
elif [ "$tutor_model" == "Qwen2-7B-Instruct" ]; then
    tutor_model_name_or_path=$model_base_path/Qwen2-7B-Instruct
elif [ "$tutor_model" == "Qwen2-72B-Instruct" ]; then
    tutor_model_name_or_path=$model_base_path/Qwen2-72B-Instruct
else
    echo "Not supported models!"
fi



student_model_name_or_path=$model_base_path/Mixtral-8x7B-Instruct-v0.1-AWQ
student_vllm_endpoint="http://localhost:8002/v1"
student_levels=(low_level med_level high_level)

verifier_base_model_path=$model_base_path/Qwen3-8B
tutor_num_responses=10

for level in ${student_levels[@]}; do
    echo "Running interactive simulation between tutor and $level student ..."
    verifier_model_dir=output/reward_model/$level
    python traver/run_traver.py --tutor_setting $tutor_setting \
        --namespace_file $namespace_file \
        --prompt_element_file $prompt_element_file \
        --output_dir $output_dir \
        --verifier_base_model_path $verifier_base_model_path \
        --verifier_model_dir $verifier_model_dir \
        --tutor_model_name_or_path $tutor_model_name_or_path \
        --tutor_num_responses $tutor_num_responses \
        --student_model_name_or_path $student_model_name_or_path \
        --student_setting $level \
        --api_key_file $api_key_file \
        --azure_endpoint $azure_endpoint \
        --vllm_endpoint_tutor $tutor_vllm_endpoint \
        --vllm_endpoint_student $student_vllm_endpoint \
        --show_description false \
        --show_message false
done