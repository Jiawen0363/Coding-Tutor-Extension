#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def count_levels(json_file):
    """统计 JSON 文件中不同 level 的数量"""
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计所有 parsed_level
    level_counts = Counter()
    total_records = 0
    total_conversations = len(data)
    
    for conv in data:
        if 'level_monitor_history' in conv and conv['level_monitor_history']:
            for record in conv['level_monitor_history']:
                if 'parsed_level' in record:
                    level_counts[record['parsed_level']] += 1
                    total_records += 1
    
    # 打印结果
    print("=" * 60)
    print(f"文件: {os.path.basename(os.path.dirname(json_file))}")
    print("=" * 60)
    print(f"\n总对话数: {total_conversations}")
    print(f"总监控记录数: {total_records}")
    print("\n水平分布:")
    print("-" * 40)
    
    for level in ['low', 'medium', 'high']:
        count = level_counts.get(level, 0)
        percentage = (count / total_records * 100) if total_records > 0 else 0
        print(f"  {level.upper():8s}: {count:4d} ({percentage:5.1f}%)")
    
    print("=" * 60)
    
    # 返回统计结果
    return level_counts, total_records


def analyze_and_plot(base_dir, output_file=None):
    """分析三个 student level 文件并画图"""
    
    student_levels = ['low_level', 'med_level', 'high_level']
    all_results = {}
    
    # 分析每个 student level
    for student_level in student_levels:
        json_file = os.path.join(base_dir, student_level, 'simulated_dialogs.json')
        if not os.path.exists(json_file):
            print(f"⚠️ 文件不存在: {json_file}")
            continue
        
        print(f"\n分析 {student_level}...")
        level_counts, total_records = count_levels(json_file)
        all_results[student_level] = {
            'counts': level_counts,
            'total': total_records
        }
    
    # 绘图
    if len(all_results) > 0:
        plot_results(all_results, base_dir, output_file)


def plot_results(all_results, base_dir, output_file=None):
    """绘制对比图"""
    
    student_levels = ['low_level', 'med_level', 'high_level']
    level_categories = ['low', 'medium', 'high']
    
    # 准备数据
    data_matrix = []
    for student_level in student_levels:
        if student_level in all_results:
            result = all_results[student_level]
            percentages = []
            for level_cat in level_categories:
                count = result['counts'].get(level_cat, 0)
                total = result['total']
                percentage = (count / total * 100) if total > 0 else 0
                percentages.append(percentage)
            data_matrix.append(percentages)
        else:
            data_matrix.append([0, 0, 0])
    
    data_matrix = np.array(data_matrix)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(student_levels))
    width = 0.25
    
    colors = ['#FF6B6B', '#FFA500', '#4ECDC4']  # Red, Orange, Teal
    
    for i, level_cat in enumerate(level_categories):
        offset = width * (i - 1)
        bars = ax.bar(x + offset, data_matrix[:, i], width, 
                      label=f'Predicted {level_cat.capitalize()}',
                      color=colors[i], alpha=0.8)
        
        # 添加数值标签
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Ground Truth Student Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Student Level Prediction Distribution\n{os.path.basename(base_dir)}', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Low Level', 'Medium Level', 'High Level'])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # 保存图片
    if output_file is None:
        output_file = os.path.join(base_dir, 'level_distribution.png')
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图片已保存到: {output_file}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        analyze_and_plot(base_dir, output_file)
    else:
        # 默认路径
        base_dir = "/home/wangjian/Coding-Tutor-Extension/output/dialogue/vanilla/first_iter/Qwen3-4B_MoE"
        analyze_and_plot(base_dir)

