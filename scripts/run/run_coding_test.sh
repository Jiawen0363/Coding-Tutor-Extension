# !/bin/bash
ROOT=/home/wangjian/EvoCodeBench/EvoCodeBench-2403  # Note: EvoCodeBench-2403 with configured envs
Source_Code_Root=$ROOT/Source_Code
Dependency_Root=$ROOT/Dependency_Data
metadata_file=$ROOT/metadata.jsonl

prompt_element_file=prompt/prompt_elements_final.jsonl
prompt_base_dir=prompt/student_posttest
output_base_dir=output/student_posttest

tutor_settings=(vanilla)    # TODO: change to your own setting
models=(please_merged_model) # TODO: change to your own model
student_levels=(low_level med_level high_level)
k="1,3,5,10"
n=10

log_file="output/student_posttest/test_progress.log"
echo "Starting coding test at $(date)" > $log_file

for setting in ${tutor_settings[@]}; do
    for model in ${models[@]}; do
        for level in ${student_levels[@]}; do
            for round in {1..8}; do
                echo "Running recall@$k for tutor_setting: $setting model: $model student_level: $level round: $round" | tee -a $log_file
            python traver/parser/recall_k.py \
                    --output_file $output_base_dir/$setting/$model/$level/round_$round/completion.jsonl \
                    --log_file $output_base_dir/$setting/$model/$level/round_$round/dependency_results.jsonl \
                --data_file $metadata_file \
                --source_code_root $Source_Code_Root \
                --dependency_data_root $Dependency_Root \
                --k $k
                echo "Completed recall@$k for $setting/$model/$level/round_$round at $(date)" | tee -a $log_file
            done
            echo "Running pass@$k for tutor_setting: $setting model: $model student_level: $level" | tee -a $log_file
            python traver/utils/check_source_code.py $Source_Code_Root
            python traver/parser/pass_k.py \
                --output_file $output_base_dir/$setting/$model/$level/completion.jsonl \
                --log_file $output_base_dir/$setting/$model/$level/test_results.jsonl \
                --data_file $metadata_file \
                --source_code_root $Source_Code_Root \
                --k $k \
                --n $n
            echo "Completed pass@$k for $setting/$model/$level at $(date)" | tee -a $log_file
        done
    done
done

echo "All tests completed at $(date)" | tee -a $log_file

