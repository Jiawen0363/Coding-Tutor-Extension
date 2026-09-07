#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
import torch
import argparse
from tqdm import tqdm
from verifier.model_utils import load_model, load_model_for_namespace, get_part_for_namespace
from verifier.data_utils import process_dialogue_for_namespace, OfflineDataBuilder, build_model_data, load_json_data

# python traver/testing3.py


# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from verifier.data_utils import OfflineDataBuilder, build_model_data, load_json_data

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='对话评分脚本')
    
    # 必需参数
    parser.add_argument('--dialog_file', type=str, required=True,
                       help='对话数据文件路径')
    parser.add_argument('--namespaces_file', type=str, required=True,
                       help='namespace映射文件路径')
    parser.add_argument('--output_file', type=str, required=True,
                       help='输出文件路径')
    
    # 可选参数（有默认值）
    parser.add_argument('--verifier_base_model_path', type=str, 
                       default="/data/models/Mistral-7B-v0.1",
                       help='verifier基础模型路径')
    parser.add_argument('--verifier_model_dir', type=str,
                       default="/home/wangjian/Coding-Tutor-Extension/Verifier-7B",
                       help='verifier模型目录')
    parser.add_argument('--elements_file', type=str,
                       default="/home/wangjian/Coding-Tutor-Extension/prompt/prompt_elements_final.jsonl",
                       help='prompt elements文件路径')
    parser.add_argument('--template_file', type=str,
                       default="/home/wangjian/Coding-Tutor-Extension/prompt/template/verifier.txt",
                       help='verifier模板文件路径')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 1. 加载数据及verifier模型
    print("📊 开始加载数据...")

    # 加载对话数据 - 这是标准JSON格式，不是JSONL
    with open(args.dialog_file, 'r') as f:
        dialogues = json.load(f)
    print("there are", len(dialogues), "dialogues")

    # 配置路径
    verifier_base_model_path = args.verifier_base_model_path
    verifier_model_dir = args.verifier_model_dir

    # 加载namespace映射
    with open(args.namespaces_file, 'r') as f:
        namespaces_data = json.load(f)
    part_lists = namespaces_data["part_lists"]
    print(f"📋 加载了 {len(part_lists)} 个part的namespace映射")

    # 加载prompt elements 
    elements = load_json_data(args.elements_file)

    # 加载verifier模板
    with open(args.template_file, 'r') as f:
        verifier_template = f.read()

    # 2. 按part分组对话
    print("🔀 按part分组对话...")
    dialogues_by_part = {i: [] for i in range(len(part_lists))}
    
    for dialogue_idx, dialogue in enumerate(dialogues):
        namespace = dialogue["namespace"]
        try:
            part_idx = get_part_for_namespace(namespace, part_lists)
            dialogues_by_part[part_idx].append((dialogue_idx, dialogue))
        except ValueError as e:
            print(f"⚠️ 警告: 对话 {dialogue_idx} 的 namespace '{namespace}' 未找到对应的part，跳过")
            continue
    
    # 打印分组统计
    for part_idx, dialogues_in_part in dialogues_by_part.items():
        print(f"  Part {part_idx}: {len(dialogues_in_part)} 个对话")

    # 3. 主循环：按part处理
    print("🚀 开始处理所有对话...")

    all_results = []

    # 外层循环：遍历每个part
    for part_idx in range(len(part_lists)):
        dialogues_in_part = dialogues_by_part[part_idx]
        
        if len(dialogues_in_part) == 0:
            print(f"\n⏭️  Part {part_idx}: 没有对话，跳过")
            continue
        
        print(f"\n{'='*60}")
        print(f"📦 开始处理 Part {part_idx} (共 {len(dialogues_in_part)} 个对话)")
        print(f"{'='*60}")
        
        # 加载该part的模型
        model_path = os.path.join(verifier_model_dir, f"part{part_idx}", "pytorch_model.bin")
        
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}，跳过该part")
            continue
        
        print(f"📥 加载 Part {part_idx} 的verifier模型: {model_path}")
        try:
            verifier_model, verifier_tokenizer = load_model(
                base_model_name_or_path=verifier_base_model_path,
                trained_verifier_model_path=model_path
            )
            print(f"✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}，跳过该part")
            continue
        
        try:
            # 内层循环：处理该part的所有对话
            for dialogue_idx, dialogue in tqdm(dialogues_in_part, desc=f"Part {part_idx}"):
                namespace = dialogue["namespace"]
                
                try:
                    # 调用处理函数（现在传入model和tokenizer）
                    result = process_dialogue_for_namespace(
                        dialogue=dialogue,
                        namespace=namespace,
                        model=verifier_model,
                        tokenizer=verifier_tokenizer,
                        elements=elements,
                        template=verifier_template
                    )
                    
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"❌ 对话 {dialogue_idx + 1} 处理失败: {e}")
                    print("🚫 终止进程")
                    raise e
        
        finally:
            # 清理该part的模型
            print(f"🧹 释放 Part {part_idx} 的模型内存...")
            del verifier_model, verifier_tokenizer
            torch.cuda.empty_cache()
            print(f"✅ Part {part_idx} 处理完成\n")

    # 保存所有结果
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir:  # 如果有目录路径
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✅ 所有对话处理完成，结果保存到 {args.output_file}")
    print(f"📊 共处理了 {len(all_results)} 个对话")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

