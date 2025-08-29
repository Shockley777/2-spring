#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DATASET6完整处理流程
依次执行：1. 图像分割 -> 2. 特征提取 -> 3. CSV合并
"""

import subprocess
import sys
import os
import argparse

def run_script(script_path, script_name):
    """
    运行指定的Python脚本
    """
    print(f"\n{'='*60}")
    print(f"开始运行: {script_name}")
    print(f"{'='*60}")
    
    try:
        # 使用subprocess运行脚本
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.path.dirname(script_path))
        
        # 打印输出
        if result.stdout:
            print("输出:")
            print(result.stdout)
        
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✓ {script_name} 执行成功")
            return True
        else:
            print(f"✗ {script_name} 执行失败，返回码: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"✗ 运行 {script_name} 时出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="运行完整处理流程：整理 -> 分割 -> 特征 -> 合并")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据根目录（包含各 DAY* 子文件夹）。若不提供则使用相对路径 ../data",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="子步骤失败时自动继续（非交互）",
    )
    args = parser.parse_args()

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 定义脚本路径和名称
    scripts = [
        (os.path.join(current_dir, "0organize_images.py"), "图片整理 (0organize_images.py)"),
        (os.path.join(current_dir, "1seg_triangle_V6.py"), "图像分割 (1seg_triangle_V6.py)"),
        (os.path.join(current_dir, "2featureExtract_circularity_V6.py"), "特征提取 (2featureExtract_circularity_V6.py)"),
        (os.path.join(current_dir, "3combineCsv_V6.py"), "CSV合并 (3combineCsv_V6.py)")
    ]

    print("完整处理流程启动")
    print("包含以下步骤:")
    for i, (_, name) in enumerate(scripts, 1):
        print(f"  {i}. {name}")

    # 逐个执行脚本
    success_count = 0
    for script_path, script_name in scripts:
        if os.path.exists(script_path):
            # 通过环境变量传递 data-root，子脚本会优先读取 CLI，然后是 DATA_ROOT
            env = os.environ.copy()
            if args.data_root:
                env['DATA_ROOT'] = args.data_root

            print(f"\n>>> 运行 {script_name}  (DATA_ROOT={env.get('DATA_ROOT', '')})")
            result = subprocess.run([sys.executable, script_path] + (["--data-root", args.data_root] if args.data_root else []),
                                    capture_output=True,
                                    text=True,
                                    cwd=os.path.dirname(script_path),
                                    env=env)

            if result.stdout:
                print("输出:")
                print(result.stdout)
            if result.stderr:
                print("错误信息:")
                print(result.stderr)

            if result.returncode == 0:
                print(f"✓ {script_name} 执行成功")
                success_count += 1
            else:
                print(f"✗ {script_name} 执行失败，返回码: {result.returncode}")
                if args.yes:
                    print("--yes 指定，自动继续下一步")
                else:
                    user_input = input("输入 'y' 继续，任意其他键退出: ").lower()
                    if user_input != 'y':
                        print("处理流程中断")
                        break
        else:
            print(f"✗ 脚本文件不存在: {script_path}")

    # 总结
    print(f"\n{'='*60}")
    print("处理流程完成")
    print(f"成功执行: {success_count}/{len(scripts)} 个步骤")
    print(f"{'='*60}")

    if success_count == len(scripts):
        print("🎉 所有步骤执行成功！")
        print("\n处理结果:")
        print("- 原始图片已整理到各DAY文件夹的 images/ 子目录")
        print("- 分割掩码保存在各DAY文件夹的 masks/ 子目录")
        print("- 特征文件保存在各DAY文件夹的 features/ 子目录")
        print("- 合并的CSV文件保存在各DAY文件夹的 total/ 子目录")
    else:
        print("⚠️  部分步骤执行失败，请检查错误信息")

if __name__ == "__main__":
    main()
