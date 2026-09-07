#!/usr/bin/env python3
"""合并 high/med/low_level 下的 SFT JSON 为单个文件。

默认保持兼容旧用法（Qwen3-8B 路径/文件名），也支持通过参数指定：
- --base: 三个 level 目录的父目录
- --input-name: 每个 level 下的输入文件名（JSON 数组）
- --output-name: 合并后的输出文件名（写到 base 下）
- --levels: 要合并的 level 列表
"""
import json
import os
import argparse

DEFAULT_BASE = "output/dialogue/vanilla/backbone/Qwen3-8B-sft-data"
DEFAULT_LEVELS = ["high_level", "med_level", "low_level"]
DEFAULT_INPUT_NAME = "train_imitation_Qwen3-8B.json"
DEFAULT_OUTPUT_NAME = "all_levels_combined_train_imitation_Qwen3-8B.json"


def parse_args():
    p = argparse.ArgumentParser(description="合并不同 level 的 SFT JSON（数组）为单个 JSON（数组）")
    p.add_argument("--base", default=DEFAULT_BASE, help="high/med/low_level 的父目录")
    p.add_argument("--input-name", default=DEFAULT_INPUT_NAME, help="每个 level 目录下的输入文件名")
    p.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME, help="输出文件名（写入 base 目录）")
    p.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS, help="要合并的 level 列表")
    return p.parse_args()

def main():
    args = parse_args()
    merged = []
    for level in args.levels:
        path = os.path.join(args.base, level, args.input_name)
        if not os.path.exists(path):
            print(f"跳过（不存在）: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"输入文件不是 JSON 数组: {path}")
        merged.extend(data)
        print(f"  {level}: {len(data)} 条")
    out_path = os.path.join(args.base, args.output_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"合并完成: {out_path}，共 {len(merged)} 条")

if __name__ == "__main__":
    main()
