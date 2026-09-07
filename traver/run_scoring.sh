#!/bin/bash
base_dir="/data/wangjian/Coding-Tutor-Extension"
export CUDA_VISIBLE_DEVICES=0

# 让 PyTorch 能找到 libcusparseLt.so.0（CUDA 库）
# 若已知路径，可先执行: export CUDA_LIB_PATH=/path/to/dir/containing/libcusparseLt
_cuda_lib=""
if [ -n "${CUDA_LIB_PATH:-}" ] && { [ -f "${CUDA_LIB_PATH}/libcusparseLt.so.0" ] || [ -f "${CUDA_LIB_PATH}/libcusparseLt.so" ]; }; then
  _cuda_lib="$CUDA_LIB_PATH"
fi
if [ -z "$_cuda_lib" ]; then
  for _d in /usr/local/cuda/lib64 /usr/local/cuda-12/lib64 /usr/local/cuda-11/lib64 "${CONDA_PREFIX:-}/lib" "${CONDA_PREFIX:-}/lib64"; do
    if [ -n "$_d" ] && { [ -f "${_d}/libcusparseLt.so.0" ] || [ -f "${_d}/libcusparseLt.so" ]; } 2>/dev/null; then
      _cuda_lib="$_d"
      break
    fi
  done
fi
if [ -z "$_cuda_lib" ]; then
  _search="/usr/local /usr"
  [ -n "${CONDA_PREFIX:-}" ] && _search="${CONDA_PREFIX} $_search"
  _found=$(find $_search -name "libcusparseLt.so*" -type f 2>/dev/null | head -1)
  [ -n "$_found" ] && _cuda_lib="$(dirname "$_found")"
fi
if [ -n "$_cuda_lib" ]; then
  export LD_LIBRARY_PATH="${_cuda_lib}:${LD_LIBRARY_PATH}"
  echo "使用 CUDA 库路径: $_cuda_lib"
else
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
  echo "未找到 libcusparseLt.so，已使用默认路径，若仍报错请手动设置 LD_LIBRARY_PATH"
fi

VERIFIER_BASE_MODEL_PATH="/data_old/models/Qwen3-8B"
# reward model 按 level 分目录
ELEMENTS_FILE="$base_dir/prompt/prompt_elements_final.jsonl"
TEMPLATE_FILE="$base_dir/prompt/template/verifier.txt"

TUTOR_SETTING="vanilla"
TUTOR_MODEL="Qwen3-8B"
ITERATION="backbone"

STUDENT_LEVEL=("low_level" "med_level" "high_level")
SCORING_FAILED=0
for level in "${STUDENT_LEVEL[@]}"; do
    VERIFIER_MODEL_DIR="$base_dir/output/reward_model/$level"
    DIALOG_FILE="$base_dir/output/dialogue/$TUTOR_SETTING/$ITERATION/$TUTOR_MODEL/$level/simulated_dialogs.json"
    NAMESPACES_FILE="$base_dir/prompt/namespaces.json"
    OUTPUT_FILE="$base_dir/output/scored/$TUTOR_SETTING/$ITERATION/$TUTOR_MODEL/$level/simulated_dialogs_scoring.json"
    python3 traver/run_scoring.py \
        --dialog_file "$DIALOG_FILE" \
        --namespaces_file "$NAMESPACES_FILE" \
        --output_file "$OUTPUT_FILE" \
        --verifier_base_model_path "$VERIFIER_BASE_MODEL_PATH" \
        --verifier_model_dir "$VERIFIER_MODEL_DIR" \
        --elements_file "$ELEMENTS_FILE" \
        --template_file "$TEMPLATE_FILE" || SCORING_FAILED=1
done

if [ "$SCORING_FAILED" -eq 0 ]; then
    bash traver/TTT/format_convert.sh "$TUTOR_MODEL"
    echo ""
    echo "✅ 脚本执行完成!"
else
    echo ""
    echo "❌ 打分阶段失败，已跳过格式转换。"
    exit 1
fi 