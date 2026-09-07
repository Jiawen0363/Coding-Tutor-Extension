#!/usr/bin/env python3
"""
数据格式转换脚本：将 query-target 格式转换为 Alpaca 格式
"""

import json
import os
from pathlib import Path

def convert_to_alpaca(input_file, output_file):
    """
    将 query-target 格式转换为 Alpaca 格式
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    print(f"正在读取文件: {input_file}")
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据: {len(data)} 个样本")
    
    # 转换为Alpaca格式
    converted_data = []
    for i, item in enumerate(data):
        converted_item = {
            'instruction': item['query'],
            'input': '',
            'output': item['target']
        }
        converted_data.append(converted_item)
        
        # 显示进度
        if (i + 1) % 100 == 0:
            print(f"已处理: {i + 1}/{len(data)} 个样本")
    
    # 保存转换后的数据
    print(f"正在保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！共 {len(converted_data)} 个样本")
    print(f"输出文件: {output_file}")
    
    # 显示转换后的格式示例
    print("\n转换后的格式示例:")
    print("=" * 50)
    print(json.dumps(converted_data[0], indent=2, ensure_ascii=False)[:500])
    print("...")

def main():
    # 输入和输出文件路径
    input_file = "/code/Coding_Tutor_Extension/output/dialogue/vanilla/gpt-4o/all_levels_combined.json"
    output_file = "/code/Coding_Tutor_Extension/output/dialogue/vanilla/gpt-4o/all_levels_combined_alpaca.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 执行转换
    convert_to_alpaca(input_file, output_file)

if __name__ == "__main__":
    main()