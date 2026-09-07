# -*- coding: utf-8 -*-

#告诉python在哪里找我的utils包
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)



import time
import json
import argparse
import tiktoken
from tqdm import tqdm
from chatarena.agent import Player, Moderator
from chatarena.agent_moetutor import MonitoredTutor
from chatarena.backends import GPTChat, O1Chat, VLLMChat
from chatarena.environments.conversation_tutoring import TutoringConversation
from chatarena.arena_tutoring import TutoringArena
from utils.utils import (
    load_json_data, 
    load_api, 
    load_finished_data, 
    convert_to_json
)
from utils.make_prompt import (
    prompt_student, 
    prompt_tutor, 
    prompt_moderator
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tutor_setting", type=str, default="vanilla")
    parser.add_argument("--prompt_element_file", type=str, default='prompt/prompt_elements_final.jsonl')
    parser.add_argument("--output_dir", type=str,  required=True,
                        help="The output directory to save the simulated dialog data.")
    
    # Level monitoring parameters
    parser.add_argument("--level_monitor_endpoint", type=str, default="http://localhost:8003/v1",
                        help="VLLM endpoint for level monitoring model (Qwen3-4B).")
    parser.add_argument("--level_monitor_model", type=str, default="/data/models/Qwen3-4B",
                        help="Model path for level monitoring.")
    
    # Adapter names for different levels
    parser.add_argument("--low_level_adapter", type=str, default="low_level_tutor",
                        help="Adapter name for low-level students.")
    parser.add_argument("--medium_level_adapter", type=str, default="medium_level_tutor",
                        help="Adapter name for medium-level students.")
    parser.add_argument("--high_level_adapter", type=str, default="high_level_tutor",
                        help="Adapter name for high-level students.")
    
    parser.add_argument("--tutor_model_name_or_path", type=str, default="gpt-4o")
    parser.add_argument("--tutor_max_tokens", type=int, default=300, 
                        help="The max number of tokens to generate for the tutor.")
    
    parser.add_argument("--student_model_name_or_path", type=str, default="Mixtral-8x7B-Instruct")
    parser.add_argument("--student_setting", type=str, choices=['low_level', 'med_level', 'high_level'])
    parser.add_argument("--student_max_tokens", type=int, default=300,
                        help="The max number of tokens to generate for the student.")
    parser.add_argument("--max_code_context", type=int, default=1024,
                        help="The max number of tokens for context above the target code.")
    
    parser.add_argument('--api_key_file', type=str, default=None)
    parser.add_argument('--azure_endpoint', type=str, default=None)
    parser.add_argument('--vllm_api_key', type=str, default='EMPTY')
    parser.add_argument('--vllm_endpoint_tutor', type=str, default='http://localhost:8001/v1')
    parser.add_argument('--vllm_endpoint_student', type=str, default='http://localhost:8002/v1')
    parser.add_argument("--max_interaction_round", type=int,default=8, 
                        help="The max number of interaction rounds.")
    parser.add_argument("--max_latest_messages", type=int, default=8,
                        help="The maximum number of latest messages to consider in the backend.")
    parser.add_argument("--temperature", type=float, default=0.4, 
                        help="The temperature to use in sampling.")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="The top_p to use in sampling.")
    parser.add_argument("--show_description", type=str2bool, default="true", 
                        help="Whether to show the role description.")
    parser.add_argument("--show_message", type=str2bool, default="true", 
                        help="Whether to show the conversation messages.")
    return parser.parse_args()

def str2bool(v):
    if v.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif v.lower() in ('false',' no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")

def get_output_path(args):
    if "gpt-" in args.tutor_model_name_or_path or "o1" in args.tutor_model_name_or_path:
        output_subdir = os.path.join(args.output_dir, 
                                     f"{args.tutor_setting}/{args.tutor_model_name_or_path}/{args.student_setting}")
    else:
        model_name = args.tutor_model_name_or_path.split("/")[-1]
        output_subdir = os.path.join(args.output_dir, f"{args.tutor_setting}/{model_name}/{args.student_setting}")
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    output_path = os.path.join(output_subdir, "simulated_dialogs.jsonl")
    return output_path

def interactive_simulation(args, prompt_data, tutor_backend, student_backend, moderator_backend, 
                           level_monitor_backend, prompt_elements):
    """Generate dialog data between the student and the tutor with level monitoring."""
    output_path = get_output_path(args)
    
    # load output data to be simulated (skip finished data)
    finished_data = load_finished_data(output_path)
    print(f"Skip {len(finished_data)} finished data.")
    
    # Create namespace to element mapping for quick lookup
    namespace_to_element = {elem['namespace']: elem for elem in prompt_elements}

    with open(output_path, "a", encoding='utf-8') as fw:
        for js in tqdm(prompt_data):
            if js["namespace"] in finished_data:
                continue
            
            # Get the full prompt element data for this namespace
            prompt_element = namespace_to_element[js["namespace"]]
            
            # Adapter names configuration
            adapter_names = {
                "low": args.low_level_adapter,
                "medium": args.medium_level_adapter,
                "high": args.high_level_adapter
            }
            
            tutor = MonitoredTutor(
                name="tutor",
                role_desc=js["tutor_desc"],
                backend=tutor_backend,
                level_monitor_backend=level_monitor_backend,
                prompt_element=prompt_element,
                adapter_names=adapter_names
            )

            student = Player(
                name="student", role_desc=js["student_desc"],
                backend=student_backend,
            )

            moderator = Moderator(
                role_desc=js["moderator_desc"], 
                backend=moderator_backend,
                terminal_condition="According to the dialogue history above, do you think the tutor's goal is completed? Please answer 'yes' or 'no'.",
            )

            env = TutoringConversation(
                player_names=[tutor.name, student.name], 
                moderator=moderator, 
                moderator_period="round"
            )
            
            arena = TutoringArena(players=[tutor, student], environment=env)
            arena.launch_cli(max_steps=args.max_interaction_round * 2, 
                             show_description=args.show_description, 
                             show_message=args.show_message, 
                             interactive=False)
            
            # save the simulated dialog to file
            messages = env.get_observation()
            simulated_convs = []
            for msg in messages:
                if msg.agent_name == tutor.name:
                    utt = {"tutor": msg.content}
                else:
                    utt = {"student": msg.content}
                simulated_convs.append(utt)
            
            # Get level monitor history from tutor
            level_monitor_data = tutor.level_monitor_history
            
            write_line = {
                "namespace": js["namespace"],
                "conversation": simulated_convs,
                "level_monitor_history": level_monitor_data
            }
            fw.write(json.dumps(write_line, ensure_ascii=False) + "\n")
            fw.flush()

    print(f"Saved to {output_path}")

    # for readability, convert the jsonl file to json format
    convert_to_json(output_path, output_path.replace(".jsonl", ".json"))


def run_simulation(args, prompt_data, prompt_elements):
    print(f"Total of {len(prompt_data)} prompt samples.")

    if args.tutor_model_name_or_path in ("gpt-3.5", "gpt-4", "gpt-4o"):
        assert args.api_key_file is not None, "Please provide the API key file for OpenAI."
        api_key = load_api(args.api_key_file)
        if args.tutor_model_name_or_path == "gpt-3.5":
            tutor_model = "gpt35-1106"  # Azure OpenAI deploy name for 'gpt-3.5-turbo-1106'
        elif args.tutor_model_name_or_path == "gpt-4":
            tutor_model = 'GPT4-1106-preview' # Azure OpenAI deploy name for 'gpt-4-1106-preview'
        elif args.tutor_model_name_or_path == "gpt-4o":
            tutor_model = 'GPT4o'   # Azure OpenAI deploy name for 'gpt-4o-2024-05-13'
        else:
            raise ValueError(f"Unknown tutor model name: {args.tutor_model_name_or_path}")
        
        tutor_backend = GPTChat(
            api_key=api_key,
            azure_endpoint=args.azure_endpoint,
            model=tutor_model, 
            temperature=args.temperature, 
            top_p=args.top_p, 
            max_tokens=args.tutor_max_tokens,
            max_latest_messages=args.max_latest_messages
        )

    elif "o1" in args.tutor_model_name_or_path:
        assert args.api_key_file is not None, "Please provide the API key file for OpenAI."
        api_key = load_api(args.api_key_file)
        tutor_model = 'o1-mini'
        tutor_backend = O1Chat(
            api_key=api_key,
            azure_endpoint=args.azure_endpoint,
            model=tutor_model,
            api_version = "2024-09-01-preview",
            max_latest_messages=args.max_latest_messages
        )
    else:
        tutor_backend = VLLMChat(
            vllm_api_key=args.vllm_api_key,
            vllm_endpoint=args.vllm_endpoint_tutor,
            model_name_or_path=args.tutor_model_name_or_path,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.tutor_max_tokens,
            max_latest_messages=args.max_latest_messages
        )
    # set the `temperature` and `top_p` to fixed values for consistent comparison
    student_backend = VLLMChat(
        vllm_api_key=args.vllm_api_key,
        vllm_endpoint=args.vllm_endpoint_student,
        model_name_or_path=args.student_model_name_or_path,
        max_tokens=args.student_max_tokens,
        max_latest_messages=args.max_latest_messages,
        temperature=0.4,
        top_p=0.95
    )
    moderator_backend = VLLMChat(
        vllm_api_key=args.vllm_api_key,
        vllm_endpoint=args.vllm_endpoint_student,
        model_name_or_path=args.student_model_name_or_path,
        temperature=0.1,
        top_p=0.95,
        max_tokens=100,
        max_latest_messages=-1
    )
    
    # Level monitoring backend (Qwen3-4B)
    level_monitor_backend = VLLMChat(
        vllm_api_key=args.vllm_api_key,
        vllm_endpoint=args.level_monitor_endpoint,
        model_name_or_path=args.level_monitor_model,
        temperature=0.1,  # Low temperature for consistent output
        top_p=0.95,
        max_tokens=10,  # Only need one word
        max_latest_messages=-1
    )
    print(f"✅ Level monitoring enabled with model: {args.level_monitor_model}")
    print(f"   Endpoint: {args.level_monitor_endpoint}")

    # run interactive simulation
    interactive_simulation(args, prompt_data, tutor_backend, student_backend, moderator_backend,
                           level_monitor_backend, prompt_elements)


def main(args):
    prompt_elements = load_json_data(args.prompt_element_file)
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    # Prepare prompt data for all elements
    prompt_data_all = []
    for d in tqdm(prompt_elements):
        tutor_desc = prompt_tutor(d, tokenizer,
                                setting="base",
                                max_code_context=args.max_code_context)
        student_desc = prompt_student(d, tokenizer,
                                    level=args.student_setting,
                                    max_code_context=args.max_code_context,
                                    is_pretest=False)
        moderator_desc = prompt_moderator(d, tokenizer, max_code_context=args.max_code_context)
        prompt_data_all.append({
            "namespace": d['namespace'], 
            "tutor_desc": tutor_desc,
            "student_desc": student_desc,
            "moderator_desc": moderator_desc
        })
    
    run_simulation(args, prompt_data_all, prompt_elements)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
