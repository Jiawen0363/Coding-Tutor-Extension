#!/usr/bin/env python3

import json
import os
import sys
import argparse
import tiktoken
import random

# 添加路径以便导入 make_prompt（sft/ -> 项目根 -> traver/utils）
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, 'traver', 'utils'))
from make_prompt import get_element, prompt_tutor

sys.path.insert(0, _project_root)
from opd.privileged import student_prior_knowledge

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
        
    def _is_empty_tutor_turn(self, turn: dict) -> bool:
        """判断 tutor 轮次是否为空（空串/纯空白/None）。"""
        if not isinstance(turn, dict):
            return False
        if "tutor" not in turn:
            return False
        v = turn.get("tutor", None)
        if v is None:
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return False

    def _truncate_conversation_on_empty_tutor(self, conversation):
        """
        按用户要求清洗对话：
        - 一旦出现空 tutor（"tutor": "" / 纯空白 / None），
          删除该 tutor turn 以及其后的所有对话片段（直接截断）。
        """
        if not isinstance(conversation, list):
            return conversation, False, None
        for i, turn in enumerate(conversation):
            if self._is_empty_tutor_turn(turn):
                return conversation[:i], True, i
        return conversation, False, None

    def _get_conversation_context(self, conversation_tx, response_idx):
        """
        获取指定response_idx对应的对话上下文
        包含该tutor utterance之前的所有对话 + 该tutor utterance本身
        """
        # 假设conversation_tx是一个对话列表，每个元素是一个turn
        # 我们需要找到第response_idx个tutor utterance在对话中的位置
        tutor_count = 0
        context_end = 0
        
        for i, turn in enumerate(conversation_tx):
            if "tutor" in turn:
                if tutor_count == response_idx:
                    # 找到了对应的tutor utterance，context包含到该位置（包含该tutor utterance）
                    context_end = i + 1
                    break
                tutor_count += 1
        
        # 返回该位置之前的对话 + 该tutor utterance作为context
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
        构建 fastchat 期望的格式：{instruction: prompt, conversations: [...]}
        """
        # 1. 获取 prompt 元素
        element = get_element(self.prompt_elements, namespace)
        if element is None:
            raise ValueError(f"找不到 namespace {namespace} 对应的 prompt 元素")
        
        # 2. student / teacher prompts for OPD; keep full base instruction for legacy SFT
        instruction = prompt_tutor(element, self.tokenizer, setting="base", max_code_context=self.max_code_context)
        student_instruction = prompt_tutor(
            element, self.tokenizer, setting="student", max_code_context=self.max_code_context
        )
        privileged_context = prompt_tutor(
            element,
            self.tokenizer,
            setting="privileged",
            max_code_context=self.max_code_context,
            student_level=self.student_level,
        )
        
        # 3. 使用 _get_conversation_context 获取对话上下文
        conversation_context = self._get_conversation_context(conversation_tx, response_idx)
        
        # 4. 构建结构化的对话历史
        history = self._build_structured_history(conversation_context)
        
        # 5. 转换为 fastchat 期望的 conversations 格式
        conversations = []
        for message in history:
            conversations.append({
                "from": message["role"],  # "tutor" 或 "student"
                "value": message["content"]
            })
        
        # 6. 返回 fastchat 期望的格式，并附带 OPD/SDFT 所需的 privileged metadata
        return {
            "instruction": instruction,
            "student_instruction": student_instruction,
            "privileged_context": privileged_context,
            "student_level": self.student_level,
            "student_prior_knowledge": student_prior_knowledge(element, self.student_level),
            "namespace": namespace,
            "conversations": conversations,
        }

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
        conversation_raw = dialogue_data["conversation"]
        conversation, truncated, truncated_at = self._truncate_conversation_on_empty_tutor(conversation_raw)
        if truncated:
            print(f"🧹 namespace {namespace}：检测到空 tutor，在 turn={truncated_at} 截断对话（删除该 turn 及之后片段）")
        
        # 3. 找到所有 tutor 轮次
        tutor_indices = []
        for i, turn in enumerate(conversation):
            if isinstance(turn, dict) and "tutor" in turn and not self._is_empty_tutor_turn(turn):
                tutor_indices.append(i)
        
        if len(tutor_indices) == 0:
            print(f"警告：namespace {namespace} 的对话中没有 tutor 轮次")
            return []
        
        # 4. 为每个 tutor 轮次生成训练样本
        sft_samples = []
        for tutor_idx, turn_idx in enumerate(tutor_indices):
            try:
                # 构建 fastchat 格式的数据
                fastchat_data = self._build_query(namespace, conversation, tutor_idx)
                
                # 创建 SFT 训练样本（直接使用 fastchat 格式）
                sft_sample = fastchat_data
                
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
        
        # 保存到 JSON 文件（使用 --output_file 指定的完整路径）
        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(all_sft_samples, f, ensure_ascii=False, indent=2)
        
        print(f"✅ SFT 数据已保存到: {self.output_file}")
        
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
    