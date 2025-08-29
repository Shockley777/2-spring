#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import glob
import pandas as pd

def process_day_folder(day_folder_path):
    """
    处理单个DAY文件夹，合并其中的CSV文件
    """
    # 定义文件夹路径
    input_folder = os.path.join(day_folder_path, 'features')
    output_folder = os.path.join(day_folder_path, 'total')
    output_file = os.path.join(output_folder, 'merged.csv')

    # 先确保输出目录存在，便于用户检查
    os.makedirs(output_folder, exist_ok=True)

    # 检查features文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"Features folder not found in {day_folder_path}")
        return

    # 获取所有CSV文件的路径
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return

    # 读取所有CSV文件，并存储在列表中
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

    if not df_list:
        print(f"No valid CSV files found in {input_folder}")
        return

    # 合并所有DataFrame，忽略原有索引
    merged_df = pd.concat(df_list, ignore_index=True)

    # 新增一列表示数据序号，从1开始
    merged_df.insert(0, 'sequence', range(1, len(merged_df) + 1))

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 将合并后的DataFrame保存为CSV文件，不保存行索引
    merged_df.to_csv(output_file, index=False)

    print(f"合并后的文件已保存至: {output_file}")
    print(f"总共合并了 {len(df_list)} 个CSV文件，包含 {len(merged_df)} 条记录")

def process_all_day_folders(data_root):
    """
    处理所有DAY文件夹
    """
    print(f"开始处理目录: {data_root}")
    
    # 遍历所有 Day*/DAY* 文件夹（大小写不敏感）
    for day_folder in sorted(os.listdir(data_root)):
        if not os.path.isdir(os.path.join(data_root, day_folder)):
            continue
        if not day_folder.strip().lower().startswith('day'):
            continue
        day_path = os.path.join(data_root, day_folder)
        features_folder = os.path.join(day_path, 'features')
        print(f"\n=== 处理 {day_folder} ===")
        if os.path.exists(features_folder):
            process_day_folder(day_path)
        else:
            # 仍然创建 total 目录，便于发现结构
            os.makedirs(os.path.join(day_path, 'total'), exist_ok=True)
            print(f"跳过 {day_folder}: features文件夹不存在 -> 已创建空的 total 目录")

def main():
    parser = argparse.ArgumentParser(description="合并各 DAY*/features/*.csv 为 DAY*/total/merged.csv")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据根目录（包含各 DAY* 子文件夹）。若不提供则使用相对路径 ../data",
    )
    args = parser.parse_args()

    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    data_root = args.data_root or os.environ.get('DATA_ROOT') or default_root

    print(f"开始合并CSV文件... 数据根目录: {data_root}")
    process_all_day_folders(data_root)
    print("\n所有文件夹处理完成！")

if __name__ == '__main__':
    main()



