#!/bin/bash

# 遍历不同学生水平
for level in high_level med_level low_level; do
    echo "处理学生水平: $level"
    python sft/format_convert.py \
        --simulated_file "/data/wangjian/Coding-Tutor-Extension/validate_reward/output/dialogue/vanilla/first_iter/deepseek-v4-flash/${level}/simulated_dialogs.json" \
        --prompt_elements_file "prompt/prompt_elements_final.jsonl" \
        --output_file "/data/wangjian/Coding-Tutor-Extension/validate_reward/output/dialogue/vanilla/first_iter/deepseek-v4-flash/${level}/train_imitation_deepseek-v4-flash.json" \
        --student_level "${level}" \
        --template_file "prompt/template/tutor_base.txt" \
        --max_code_context 1024
done

echo "所有学生水平的SFT数据转换完成！"