#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
import argparse

def organize_images_in_folder(day_folder_path):
    """
    在单个DAY文件夹中创建images子文件夹，并将所有图片移动到其中
    """
    print(f"处理文件夹: {day_folder_path}")
    
    # 创建images文件夹
    images_folder = os.path.join(day_folder_path, 'images')
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
        print(f"创建images文件夹: {images_folder}")
    
    # 定义图片扩展名
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    
    # 获取文件夹中的所有文件
    moved_count = 0
    for filename in os.listdir(day_folder_path):
        if filename.lower().endswith(valid_extensions):
            src_path = os.path.join(day_folder_path, filename)
            dst_path = os.path.join(images_folder, filename)
            
            # 检查是否是文件（而不是文件夹）
            if os.path.isfile(src_path):
                # 如果目标文件已存在，跳过
                if os.path.exists(dst_path):
                    print(f"文件已存在，跳过: {filename}")
                    continue
                
                # 移动文件
                shutil.move(src_path, dst_path)
                moved_count += 1
                print(f"移动文件: {filename} -> images/{filename}")
    
    print(f"在 {os.path.basename(day_folder_path)} 中移动了 {moved_count} 个图片文件")
    return moved_count

def organize_all_day_folders(data_root):
    """
    处理所有DAY文件夹，将图片整理到images子文件夹中
    """
    print(f"开始整理目录: {data_root}")
    total_moved = 0
    
    # 遍历所有可能的 Day/DAY 文件夹（兼容 Dataset7 的 "Day 0" 命名）
    for entry_name in sorted(os.listdir(data_root)):
        entry_path = os.path.join(data_root, entry_name)
        if not os.path.isdir(entry_path):
            continue
        normalized = entry_name.strip().lower()
        if normalized.startswith('day'):
            print(f"\n=== 整理 {entry_name} ===")
            moved_count = organize_images_in_folder(entry_path)
            total_moved += moved_count
        else:
            print(f"跳过 {entry_name}: 名称不匹配 day*")
    
    print(f"\n整理完成！总共移动了 {total_moved} 个图片文件")
    return total_moved

def main():
    parser = argparse.ArgumentParser(description="将图片移动到各 DAY*/images/ 子目录")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据根目录（包含各 DAY* 子文件夹）。若不提供则使用相对路径 ../data",
    )
    args = parser.parse_args()

    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    data_root = args.data_root or os.environ.get('DATA_ROOT') or default_root

    print(f"开始整理图片文件... 数据根目录: {data_root}")
    organize_all_day_folders(data_root)
    print("\n所有图片整理完成！")

if __name__ == '__main__':
    main()
