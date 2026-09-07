
# this file is used to convert the format of the dataset

# python traver/TTT/format_convert.py


import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.make_prompt import prompt_tutor, get_element
from transformers import AutoTokenizer


class StepPPODataConverter:
    def __init__(self, prompt_elements_file, model_name_or_path):
        self.prompt_elements = self._load_prompt_elements(prompt_elements_file)
        self.tutor_template = self._load_tutor_template()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


    def build_tutor_prompt(self, namespace):
        d = get_element(self.prompt_elements, namespace)
        prompt = prompt_tutor(d, self.tokenizer)
        return prompt

    def convert_single_dialogue(self, dialogue_data):
        namespace = dialogue_data['namespace']
        conversation = dialogue_data['conversation']
        
        # ===== 第一步：生成 prompt_list =====
        prompt0 = self.build_tutor_prompt(namespace)
        prompt_list = [prompt0]
        
        # 遍历对话，提取 student utterance
        for i in range(len(conversation)):
            if 'student' in conversation[i]:
                prompt_list.append(conversation[i]['student'])
        
        # ===== 第二步：生成 response_list =====
        response_list = []
        for i in range(len(conversation)):
            if 'tutor' in conversation[i]:
                response_list.append(conversation[i]['tutor'])
        
        # ===== 第三步：生成 reward_list =====
        reward_list = []
        for i in range(len(conversation)):
            if 'tutor' in conversation[i]:
                reward_list.append(conversation[i].get('tutor_score', 0.0))
        
        # ===== 第四步：验证和调整 =====
        # 确保 prompt_list 长度为 response_list 长度 + 1
        if len(prompt_list) > len(response_list) + 1:
            prompt_list = prompt_list[:len(response_list) + 1]
        
        return {
            "prompt": prompt_list,
            "response": response_list,
            "reward": reward_list
        }
    
    def convert_file(self, input_file, output_file):

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for dialogue in data:
                step_ppo_item = self.convert_single_dialogue(dialogue)
                out_f.write(json.dumps(step_ppo_item, ensure_ascii=False) + '\n')

    def _load_prompt_elements(self, prompt_elements_file):
        """加载prompt_elements_final.jsonl文件"""
        data = []
        with open(prompt_elements_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
            return data

    def _load_tutor_template(self):
        """加载tutor_base.txt模板"""
        with open('prompt/template/tutor_short.txt', 'r', encoding='utf-8') as f:
            return f.read()
    




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert dialogue data to StepPPO format')
    parser.add_argument('--prompt_elements_file', type=str, default="prompt/prompt_elements_final.jsonl",
                        help='Path to prompt elements file')
    parser.add_argument('--model_name_or_path', type=str, required=True,
                        help='Model name or local path for tokenizer')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input dialogue file path')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file path')
    
    args = parser.parse_args()
    
    converter = StepPPODataConverter(args.prompt_elements_file, args.model_name_or_path)
    converter.convert_file(args.input_file, args.output_file)