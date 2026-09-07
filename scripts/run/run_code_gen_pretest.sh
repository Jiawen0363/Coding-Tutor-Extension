#!/bin/bash

# Configuration
ROOT=/data/EvoCodeBench/EvoCodeBench-2403  # TODO: change to your own path
Source_Code_Root=$ROOT/Source_Code
Dependency_Root=$ROOT/Dependency_Data

model_base_path=/data/models  # TODO: change to your own path
model_name_or_path=$model_base_path/Mixtral-8x7B-Instruct-v0.1-AWQ
prompt_element_file=prompt/prompt_elements_final.jsonl
prompt_base_dir=prompt/student_pretest
output_base_dir=output/student_pretest/Mixtral-8x7B-Instruct

student_levels=(low_level med_level high_level oracle)  # Added oracle to match run_pretest.sh
k="1,3,5,10"
n=10

# GPU configuration for parallel processing
gpu_ids=(4 5 6 7)  # Added one more GPU for oracle level

# Step 1: Create prompts for pretest
for level in ${student_levels[@]}; do
    echo "Creating prompts for pretest: student_level: $level"
    python traver/utils/make_prompt.py --student_pretest \
        --prompt_element_file $prompt_element_file \
        --output_dir $prompt_base_dir/$level \
        --student_level $level
done

# Step 2: Run code generation in parallel
for idx in {0..3}; do  # Changed to 4 levels including oracle
    level=${student_levels[$idx]}
    current_gpu=${gpu_ids[$idx]}

    echo "Running code generation for pretest: student_level: $level on GPU: $current_gpu"
    CUDA_VISIBLE_DEVICES=$current_gpu python traver/utils/LM_inference.py --model $model_name_or_path \
        --prompt_file $prompt_base_dir/$level/prompt_all_rounds.jsonl \
        --output_dir $output_base_dir/$level &
done

# Wait for all parallel jobs to complete
wait

# Step 3: Process completions
for level in ${student_levels[@]}; do
    echo "Processing completions for: student_level: $level"
    python traver/utils/process_completion.py \
        --completion_file $output_base_dir/$level/completion_lm.jsonl \
        --output_file $output_base_dir/$level/completion.jsonl
done

echo "Pretest code generation completed!"