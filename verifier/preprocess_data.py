import os
import json
import random
from argparse import ArgumentParser
from data_utils import load_json_data, build_model_data, check_adjust_posttest

random.seed(42)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--prompt_element_file", type=str, default="../prompt/prompt_elements_final.jsonl")
    parser.add_argument("--chosen_models", type=str, default="all")
    parser.add_argument("--simulation_dir", type=str, default="output/dialogue/vanilla")
    parser.add_argument("--posttest_dir", type=str, default="output/student_posttest/vanilla")
    parser.add_argument("--output_dir", type=str, default="")
    
    return parser.parse_args()

def collect_dialog_data(dialogue_fp, posttest_fp):
    # find which namespaces correctly passed the posttest
    outcome_label = {}
    with open(posttest_fp, 'r', encoding='utf-8') as f:
        for line in f:
            js = json.loads(line)
            if js["namespace"] not in outcome_label:
                outcome_label[js["namespace"]] = 0
            if js['Result'] == 'Pass':
                outcome_label[js["namespace"]] = 1
    # annotate outcome reward for each namespace
    dialogues = []
    if dialogue_fp.endswith('.jsonl'):
        with open(dialogue_fp, 'r', encoding='utf-8') as f:
            for line in f:
                conv = json.loads(line)
                conv["outcome_label"] = outcome_label[conv["namespace"]]
                dialogues.append(conv)
    elif dialogue_fp.endswith('.json'):
        with open(dialogue_fp, 'r', encoding='utf-8') as f:
            conv_list = json.load(f)
            for conv in conv_list:
                conv["outcome_label"] = outcome_label[conv["namespace"]]
                dialogues.append(conv)
    else:
        raise ValueError("Unsupported file format for dialogue_fp. Only .jsonl and .json are supported.")

    return dialogues


def process_single_level(level, chosen_models, args, template, prompt_elements):
    """Process data for a single student level"""
    print(f"\n=== Processing {level} level data ===")
    
    all_data = []
    for model in chosen_models:
        model_dir = os.path.join(args.simulation_dir, model)
        level_dir = os.path.join(model_dir, level)
        
        if not os.path.exists(level_dir):
            print(f"Warning: {level_dir} does not exist, skipping...")
            continue
            
        print(f"Processing vanilla {model} - {level} student dialogue data")
        posttest_dir = os.path.join(args.posttest_dir, model, level)
        
        # check and adjust posttest data
        max_round = check_adjust_posttest(posttest_dir)

        dialogues = collect_dialog_data(os.path.join(level_dir, "simulated_dialogs.json"),
                                        os.path.join(posttest_dir, f"round_{max_round}/test_results.jsonl"))
        for dialog in dialogues:
            data_samples = build_model_data(prompt_elements, dialog, level, template)
            all_data.extend(data_samples)
    
    print(f"Total number of raw data samples for {level}:", len(all_data))
    
    if len(all_data) == 0:
        print(f"Warning: No data found for {level} level, skipping...")
        return

    # Pass-only: reward = 1/total_turn per tutor utterance; no failed-dialog samples or negative balancing.
    balanced_data = list(all_data)
    random.shuffle(balanced_data)
    print(f"Total number of samples (pass-only, shuffled) for {level}:", len(balanced_data))

    # Create output directory for this level
    level_output_dir = os.path.join(args.output_dir, level)
    if not os.path.exists(level_output_dir):
        os.makedirs(level_output_dir)
    
    # Save combined data for this level
    with open(os.path.join(level_output_dir, "verifier_data.jsonl"), 'w') as f:
        for data in balanced_data:
            f.write(json.dumps(data) + '\n')
    print(f"All {level} data saved to", os.path.join(level_output_dir, "verifier_data.jsonl"))
    
    # Load all namespaces and split them into k folds
    with open("../prompt/namespaces.json", 'r') as f:
        namespaces = json.load(f)
    num_parts = namespaces["num_parts"]
    part_lists = namespaces["part_lists"]
    data_parts = [[] for _ in range(num_parts)]
    data_parts_label = [[] for _ in range(num_parts)]
    for data in balanced_data:
        namespace = data["namespace"]
        for i in range(num_parts):
            if namespace in part_lists[i]:
                data_parts[i].append(data)
                data_parts_label[i].append(data["label"])
                break
    
    # stat label distribution
    for i in range(num_parts):
        positive_count = 0
        for label in data_parts_label[i]:
            if label > 0:
                positive_count += 1
        print(f"{level} Part {i} positive ratio = {positive_count / len(data_parts_label[i])}")
    
    # Save each fold for this level
    for i in range(num_parts):
        with open(os.path.join(level_output_dir, f"verifier_data_part{i}.jsonl"), 'w') as f:
            for data in data_parts[i]:
                f.write(json.dumps(data) + '\n')
        print(f"{level} data part {i} saved to", os.path.join(level_output_dir, f"verifier_data_part{i}.jsonl"))


def main(args):
    # load template
    template = open(f'../prompt/template/verifier.txt', 'r').read()

    prompt_elements = load_json_data(args.prompt_element_file)

    if args.chosen_models == "all":
        chosen_models = os.listdir(args.simulation_dir)
    else:
        chosen_models = args.chosen_models.split(',')
    print("Chosen models:", chosen_models)

    # Define the three student levels
    student_levels = ["high_level", "med_level", "low_level"]
    
    # Process each level separately
    for level in student_levels:
        process_single_level(level, chosen_models, args, template, prompt_elements)
    
    print(f"\n=== All processing completed ===")
    print(f"Data for each level saved in: {args.output_dir}/{{level}}/")
    print(f"Each level contains 5 folds: verifier_data_part0.jsonl to verifier_data_part4.jsonl")


if __name__ == "__main__":
    args = parse_args()
    main(args)
