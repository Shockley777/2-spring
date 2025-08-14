#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DATASET2 相似度分析运行脚本
运行此脚本将对 DATASET2 中的所有样本进行相似度分析
"""

import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🔬 DATASET2 相似度分析开始...")
    print("=" * 50)
    
    # 检查必要的依赖
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import wasserstein_distance, pearsonr, entropy
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import MinMaxScaler
        print("✅ 所有依赖包已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请运行: pip install pandas numpy matplotlib seaborn scipy scikit-learn")
        return
    
    # 检查数据文件是否存在
    sample_path = "DAY1/data7/total/merged.csv"
    if not os.path.exists(sample_path):
        print(f"❌ 数据文件不存在: {sample_path}")
        print("请确保已经运行过 seg_feature_merge.py 生成数据文件")
        return
    
    print("✅ 数据文件检查通过")
    
    # 导入并运行相似度分析
    try:
        import similarity_analysis
        print("✅ 相似度分析模块导入成功")
        print("🚀 开始执行分析...")
        
        # 分析会自动执行，因为similarity_analysis.py中的代码在导入时就会运行
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("=" * 50)
    print("🎉 DATASET2 相似度分析完成！")
    print("📁 结果文件保存在 results/ 文件夹中")

if __name__ == "__main__":
    main() 