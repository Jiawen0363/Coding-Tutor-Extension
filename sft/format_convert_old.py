#!/usr/bin/env python3

import json
import os
import sys
import argparse
import tiktoken
import random

# 添加路径以便导入 make_prompt
sys.path.append(os.path.join(os.path.dirname(__file__), '../../traver/utils'))
from make_prompt import get_element, prompt_tutor

# 在文件顶部添加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../traver/verifier'))

def parse_args():
    parser = argparse.ArgumentParser(description="将对话数据转换为SFT训练格式")
    
    parser.add_argument("--simulated_file", type=str, required=True,
                        help="simulated_dialogs.json 文件路径")
    parser.add_argument("--prompt_elements_file", type=str, required=True,
                        help="prompt_elements_final.jsonl 文件路径")
    parser.add_argument("--output_file", type=str, required=True,
                        help="输出 SFT 数据文件路径")
    parser.add_argument("--student_level", type=str, required=True,
                        choices=['high_level', 'med_level', 'low_level'],
                        help="学生水平")
    parser.add_argument("--template_file", type=str, 
                        default="prompt/template/tutor_base.txt",
                        help="tutor 模板文件路径")
    parser.add_argument("--max_code_context", type=int, default=1024,
                        help="代码上下文最大 token 数")
    
    return parser.parse_args()


class DialogToSFTConverter:
    def __init__(self, 
                 simulated_file: str,
                 prompt_elements_file: str,
                 output_file: str,
                 student_level: str,
                 template_file: str = "prompt/template/tutor_base.txt",
                 max_code_context: int = 1024):
        """
        Args:
            simulated_file: simulated_dialogs.json 文件路径
            prompt_elements_file: prompt_elements_final.jsonl 文件路径  
            output_file: 输出 SFT 数据文件路径
            student_level: 学生水平 (high_level/med_level/low_level)
            template_file: tutor 模板文件路径
            max_code_context: 代码上下文最大 token 数
        """
        self.simulated_file = simulated_file
        self.prompt_elements_file = prompt_elements_file
        self.output_file = output_file
        self.student_level = student_level
        self.template_file = template_file
        self.max_code_context = max_code_context
        
        # 初始化 tokenizer
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        # 加载数据
        self.simulated_dialogs = self._load_simulated_dialogs()
        self.prompt_elements = self._load_prompt_elements()
        
    def _get_conversation_context(self, conversation_tx, response_idx):
        """
        获取指定response_idx对应的对话上下文
        只包含该tutor utterance之前的对话
        """
        # 假设conversation_tx是一个对话列表，每个元素是一个turn
        # 我们需要找到第response_idx个tutor utterance在对话中的位置
        tutor_count = 0
        context_end = 0
        
        for i, turn in enumerate(conversation_tx):
            if "tutor" in turn:
                if tutor_count == response_idx:
                    # 找到了对应的tutor utterance，context到此为止
                    context_end = i
                    break
                tutor_count += 1
        
        # 返回该位置之前的对话作为context
        return conversation_tx[:context_end]

    def object2str(self, object_list, random_choice=True):
        """Format a list of items in various natural ways"""
        format_choices = {
            "comma": lambda x: ", ".join(x),
            "bracket": lambda x: "[" + ", ".join(x) + "]",
            "parenthesis": lambda x: "(" + ", ".join(x) + ")",
            "curly": lambda x: "{" + ", ".join(x) + "}",
            "oxford_comma": lambda x: ", ".join(x[:-1]) + ", and " + x[-1],
            "no_oxford_comma": lambda x: ", ".join(x[:-1]) + " and " + x[-1],
            "and_separated": lambda x: " and ".join(x),
        }

        if random_choice:
            format = random.choice(list(format_choices.keys()))
            return format_choices[format](object_list)
        else:
            return {format: func(object_list) for format, func in format_choices.items()}

    def _build_structured_history(self, conversation_context):
        """构建结构化的对话历史"""
        history = []
        for turn in conversation_context:
            if isinstance(turn, dict):
                if "tutor" in turn:
                    history.append({"role": "tutor", "content": turn["tutor"]})
                elif "student" in turn:
                    history.append({"role": "student", "content": turn["student"]})
                else:
                    raise ValueError(f"Unknown utterance format: {turn}")
            else:
                raise ValueError(f"Unexpected utterance format: {turn}")
        return history

    def _build_query(self, namespace, conversation_tx, response_idx):
        """
        构建 query，包含 tutor prompt + conversation history + "tutor："
        支持多种格式和随机化
        """
        # 1. 获取 prompt 元素
        element = get_element(self.prompt_elements, namespace)
        if element is None:
            raise ValueError(f"找不到 namespace {namespace} 对应的 prompt 元素")
        
        # 2. 使用 prompt_tutor 构建基础 prompt
        base_prompt = prompt_tutor(element, self.tokenizer, setting="base", max_code_context=self.max_code_context)
        
        # 3. 使用 _get_conversation_context 获取对话上下文
        conversation_context = self._get_conversation_context(conversation_tx, response_idx)
        
        # 4. 构建结构化的对话历史
        history = self._build_structured_history(conversation_context)
        
        # 5. 随机选择格式
        dialog_prefix = random.choice([
            "\n - ", "\n### ", "\n## ", "\n# ", "\n *** ", "\n **", "\n\n"
        ])
        
        answer_str, question_str = random.choice([
            ("Tutor", "Student"),
            ("Assistant", "Human"),
        ])
        
        player_prefix = {
            "tutor": answer_str,
            "student": question_str,
        }
        
        # 6. 构建对话历史字符串
        history_str = ""
        for message in history:
            history_str += "{}{}: {}".format(
                dialog_prefix, player_prefix[message["role"]], message["content"]
            )
        
        # 7. 随机选择 prompt 类型
        prompt_type = random.choice(["chat", "chat_inverse", "alpaca"])
        
        # 添加历史对话提示语
        history_intro = random.choice([
            "Here is the conversation history:",
            "Here are the previous conversations:",
            "Previous conversation:",
            "Conversation history:",
            "Here are the previous exchanges:",
            "Here is the dialogue history:",
        ])
        
        # 初始化 query 变量
        query = ""
        
        if prompt_type in ["chat", "chat_inverse"]:
            if len(history) == 0:
                history_str = ""
                base_prompt += "The tutoring session is just starting. "
            else:
                # 在 prompt 和对话历史之间插入提示语
                history_str = "\n\n" + history_intro + "\n" + history_str
            
            if "inverse" in prompt_type:
                query = (
                    history_str
                    + base_prompt
                    + dialog_prefix
                    + player_prefix["tutor"]
                    + ": "
                )
            else:
                query = (
                    base_prompt
                    + history_str
                    + dialog_prefix
                    + player_prefix["tutor"]
                    + ": "
                )

        elif prompt_type == "alpaca":
            if len(history) == 0:
                query = base_prompt + "The tutoring session is just starting. "
            else:
                query = (
                    base_prompt
                    + "\n\n"
                    + history_intro
                    + "\n"
                    + history_str
                    + "\n\n"
                )
            
            query += (
                dialog_prefix
                + player_prefix["tutor"]
                + ": "
            )
        
        return query

    def _load_simulated_dialogs(self):
        """加载模拟对话数据"""
        with open(self.simulated_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_prompt_elements(self):
        """加载 prompt 元素数据"""
        data = []
        with open(self.prompt_elements_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
            return data

    def convert_namespace_to_sft(self, namespace):
        """
        将单个 namespace 的对话数据转换为 SFT 训练格式
        
        Args:
            namespace: 函数命名空间，如 "easyvolcap.utils.gl_utils.Quad.upload_to_texture"
        
        Returns:
            list: 该 namespace 的所有 SFT 训练样本
        """
        # 1. 找到该 namespace 对应的对话数据
        dialogue_data = None
        for dialogue in self.simulated_dialogs:
            if dialogue["namespace"] == namespace:
                dialogue_data = dialogue
                break
        
        if dialogue_data is None:
            print(f"警告：找不到 namespace {namespace} 对应的对话数据")
            return []
        
        # 2. 获取对话内容
        conversation = dialogue_data["conversation"]
        
        # 3. 找到所有 tutor 轮次
        tutor_indices = []
        for i, turn in enumerate(conversation):
            if "tutor" in turn:
                tutor_indices.append(i)
        
        if len(tutor_indices) == 0:
            print(f"警告：namespace {namespace} 的对话中没有 tutor 轮次")
            return []
        
        # 4. 为每个 tutor 轮次生成训练样本
        sft_samples = []
        for tutor_idx, turn_idx in enumerate(tutor_indices):
            try:
                # 构建 query
                query = self._build_query(namespace, conversation, tutor_idx)
                
                # 提取 target
                target = conversation[turn_idx]["tutor"]
                
                # 创建 SFT 训练样本
                sft_sample = {
                    "query": query,
                    "target": target,
                    "student_level": self.student_level
                }
                
                sft_samples.append(sft_sample)
                
            except Exception as e:
                print(f"错误：处理 namespace {namespace} 的第 {tutor_idx} 个 tutor 轮次时出错: {e}")
                continue
        
        print(f"成功处理 namespace {namespace}，生成了 {len(sft_samples)} 个训练样本")
        return sft_samples

    def convert_all_namespaces_to_sft(self):
        """
        转换所有 namespace 的对话数据为 SFT 格式
        """
        all_sft_samples = []
        
        # 获取所有唯一的 namespace
        unique_namespaces = set()
        for dialogue in self.simulated_dialogs:
            unique_namespaces.add(dialogue["namespace"])
        
        print(f"发现 {len(unique_namespaces)} 个唯一的 namespace")
        
        # 遍历每个 namespace
        for namespace in unique_namespaces:
            print(f"正在处理 namespace: {namespace}")
            namespace_samples = self.convert_namespace_to_sft(namespace)
            all_sft_samples.extend(namespace_samples)
        
        print(f"总共生成了 {len(all_sft_samples)} 个 SFT 训练样本")
        
        # 保存到 JSON 文件
        output_dir = os.path.dirname(self.output_file)
        json_filename = f"train_imitation_gpt4o.json"
        json_filepath = os.path.join(output_dir, json_filename)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为 JSON 格式
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_sft_samples, f, ensure_ascii=False, indent=2)
        
        print(f"✅ SFT 数据已保存到: {json_filepath}")
        
        return all_sft_samples

# def test_build_query():
#     """测试 _build_query 方法"""
    
#     # 创建转换器实例
#     converter = DialogToSFTConverter(
#         simulated_file="output/dialogue/vanilla/gpt-4o/high_level/simulated_dialogs.json",
#         prompt_elements_file="prompt/prompt_elements_final.jsonl",
#         output_file="output/sft_data/high_level/sft_data.jsonl",
#         student_level="high_level"
#     )
    
#     # 测试数据
#     namespace = "easyvolcap.utils.gl_utils.Quad.upload_to_texture"
#     conversation_tx = [dialog["conversation"] for dialog in converter._load_simulated_dialogs() if dialog["namespace"] == namespace][0]
#     response_idx = 2  # 第1个 tutor utterance
    
#     try:
#         # 测试 _build_query
#         query = converter._build_query(namespace, conversation_tx, response_idx)
#         print("====conversation_tx====")
#         print(conversation_tx)
#         print("====query====")
#         print(query)
#         print("====end====")
#         print(f"Query 长度: {len(query)} 字符")

#     except Exception as e:
#         print(f"❌ _build_query 测试失败: {e}")
#         raise

# def test_convert_namespace_to_sft():
#     """测试 convert_namespace_to_sft 方法"""
    
#     # 测试参数
#     simulated_file = "output/dialogue/vanilla/gpt-4o/high_level/simulated_dialogs.json"
#     prompt_elements_file = "prompt/prompt_elements_final.jsonl"
#     output_file = "output/sft_data/high_level/sft_data.jsonl"
#     student_level = "high_level"
    
#     try:
#         # 创建转换器
#         converter = DialogToSFTConverter(
#             simulated_file=simulated_file,
#             prompt_elements_file=prompt_elements_file,
#             output_file=output_file,
#             student_level=student_level
#         )
        
#         print("✅ DialogToSFTConverter 初始化成功！")
        
#         # 获取第一个 namespace 进行测试
#         if len(converter.simulated_dialogs) > 0:
#             test_namespace = converter.simulated_dialogs[0]["namespace"]
#             print(f"🧪 测试 namespace: {test_namespace}")
            
#             # 调用测试函数
#             sft_samples = converter.convert_namespace_to_sft(test_namespace)
#             print("====sft_samples====")
#             print(sft_samples)
#             # 保存sft_samples到jsonl文件
#             with open('test_sft_samples.json', 'w', encoding='utf-8') as f:
#                 for sample in sft_samples:
#                     f.write(json.dumps(sample, ensure_ascii=False) + '\n')
#             print("====end====")
                
#         else:
#             print("❌ 没有找到对话数据")
            
#     except Exception as e:
#         print(f"❌ 测试失败: {e}")
#         import traceback
#         traceback.print_exc()

if __name__ == "__main__":
    args = parse_args()
    
    print(f"🚀 开始转换 SFT 数据...")
    print(f"📁 输入文件: {args.simulated_file}")
    print(f"📁 Prompt 元素文件: {args.prompt_elements_file}")
    print(f"📁 输出目录: {os.path.dirname(args.output_file)}")
    print(f"👨‍🎓 学生水平: {args.student_level}")
    
    try:
        # 创建转换器
        converter = DialogToSFTConverter(
            simulated_file=args.simulated_file,
            prompt_elements_file=args.prompt_elements_file,
            output_file=args.output_file,
            student_level=args.student_level,
            template_file=args.template_file,
            max_code_context=args.max_code_context
        )
        
        print("✅ 转换器初始化成功！")
        
        # 执行转换
        converter.convert_all_namespaces_to_sft()
        
        print("🎉 SFT 数据转换完成！")
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    