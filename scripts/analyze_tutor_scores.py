#!/usr/bin/env python3
"""
统计各个level的tutor_score
"""
import json
import os
from pathlib import Path
from typing import Dict, List


def calculate_statistics(scores: List[float]) -> Dict:
    """计算基本统计信息"""
    if not scores:
        return {
            "count": 0,
            "mean": 0,
            "median": 0,
            "std": 0,
            "min": 0,
            "max": 0,
            "sum": 0
        }
    
    scores_sorted = sorted(scores)
    n = len(scores)
    mean = sum(scores) / n
    
    # 计算中位数
    if n % 2 == 0:
        median = (scores_sorted[n//2 - 1] + scores_sorted[n//2]) / 2
    else:
        median = scores_sorted[n//2]
    
    # 计算标准差
    variance = sum((x - mean) ** 2 for x in scores) / n
    std = variance ** 0.5
    
    return {
        "count": n,
        "mean": mean,
        "median": median,
        "std": std,
        "min": min(scores),
        "max": max(scores),
        "sum": sum(scores)
    }


def extract_tutor_scores(json_file: Path) -> List[float]:
    """从JSON文件中提取所有tutor_score"""
    scores = []
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for dialog in data:
            conversation = dialog.get('conversation', [])
            for turn in conversation:
                if 'tutor_score' in turn:
                    scores.append(turn['tutor_score'])
    
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
    
    return scores


def analyze_folder(folder_path: Path) -> Dict:
    """分析指定文件夹中的tutor scores"""
    results = {}
    
    # 检查文件夹是否存在
    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return results
    
    # 遍历所有子文件夹
    for level_dir in sorted(folder_path.iterdir()):
        if level_dir.is_dir():
            level_name = level_dir.name
            json_file = level_dir / "simulated_dialogs_scoring.json"
            
            if json_file.exists():
                print(f"\nProcessing {level_name}...")
                scores = extract_tutor_scores(json_file)
                stats = calculate_statistics(scores)
                results[level_name] = stats
            else:
                print(f"Warning: {json_file} not found")
    
    return results


def print_results(results: Dict):
    """打印统计结果"""
    print("\n" + "="*80)
    print("Tutor Score Statistics by Level")
    print("="*80)
    
    for level_name, stats in results.items():
        print(f"\n{level_name.upper()}:")
        print(f"  Total Scores: {stats['count']}")
        print(f"  Mean:         {stats['mean']:.6f}")
        print(f"  Median:       {stats['median']:.6f}")
        print(f"  Std Dev:      {stats['std']:.6f}")
        print(f"  Min:          {stats['min']:.6f}")
        print(f"  Max:          {stats['max']:.6f}")
        print(f"  Sum:          {stats['sum']:.6f}")
    
    # 计算总体统计
    all_counts = sum(stats['count'] for stats in results.values())
    all_sums = sum(stats['sum'] for stats in results.values())
    
    if all_counts > 0:
        overall_mean = all_sums / all_counts
        print(f"\nOVERALL:")
        print(f"  Total Scores: {all_counts}")
        print(f"  Overall Mean: {overall_mean:.6f}")
    
    print("\n" + "="*80)


def main():
    """主函数"""
    import sys
    
    # 如果提供命令行参数，使用参数作为文件夹路径
    if len(sys.argv) > 1:
        folder_path = Path(sys.argv[1])
    else:
        # 默认路径
        folder_path = Path("/home/wangjian/Coding-Tutor-Extension/output/scored/vanilla/first_iter/Qwen3-4B_MoE_prompt3")
    
    print(f"Analyzing folder: {folder_path}")
    results = analyze_folder(folder_path)
    print_results(results)
    
    # 可选：保存结果到JSON文件
    output_file = folder_path / "tutor_score_statistics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

