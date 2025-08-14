import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, pearsonr, entropy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import gaussian_filter1d
import os
from matplotlib.cm import get_cmap
import shutil
import random

# -------- 参数配置 --------
days = [f"DAY{i}" for i in range(2, 8)]  # DATASET2: DAY2-DAY7
data_folders = [f"data{j}" for j in range(1, 7)]  # DATASET2: data1-data6
all_refs = [(d, f) for d in days for f in data_folders]
AREA_COL = "area"
RATIO_OTHER = 1  # 其他样本全部数据
BINS = 60
RANGE = (500, 3500)
TOP_K = 5

# 随机选取5个不同的参考样本
random.seed(42)
random_refs = random.sample(all_refs, 5)

for idx, (REFERENCE_FOLDER, REFERENCE_DATA) in enumerate(random_refs, 1):
    RATIO_REF = 0.3
    result_dir = os.path.join('results', f'random_ref_{idx}_{REFERENCE_FOLDER}_{REFERENCE_DATA}')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    print(f"\n===== 第{idx}组参考样本: {REFERENCE_FOLDER}_{REFERENCE_DATA} (30%数据) =====")

    # -------- 相似度函数 --------
    def compute_histogram_similarity(hist1, hist2, method='intersection'):
        hist1 = np.asarray(hist1, dtype=np.float64) + 1e-10
        hist2 = np.asarray(hist2, dtype=np.float64) + 1e-10
        if method == 'intersection':
            return np.sum(np.minimum(hist1, hist2)) / np.sum(np.maximum(hist1, hist2))
        elif method == 'kl':
            return entropy(hist1, hist2)
        else:
            raise ValueError("Unsupported similarity method")

    # -------- 加载与采样 --------
    def load_histogram(folder, data_subfolder, ratio):
        csv_path = os.path.join(folder, data_subfolder, "total", "merged.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping.")
            return None, None
        df = pd.read_csv(csv_path)
        
        # 仅保留面积在指定范围内的数据
        area_data = df[AREA_COL].dropna()
        area_data = area_data[(area_data >= RANGE[0]) & (area_data <= RANGE[1])].values

        # 如果数据不足则跳过
        if len(area_data) < 5:
            print(f"Warning: Too few area values in range for {folder}/{data_subfolder}, skipping.")
            return None, None

        np.random.shuffle(area_data)
        sampled = area_data[:int(len(area_data) * ratio)]

        bins = np.linspace(RANGE[0], RANGE[1], BINS + 1)
        hist, _ = np.histogram(sampled, bins=bins, density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        return hist, bin_centers

    # -------- 主执行逻辑 --------
    np.random.seed(42)

    # 1. 加载参考直方图
    ref_hist, bin_centers = load_histogram(REFERENCE_FOLDER, REFERENCE_DATA, RATIO_REF)
    if ref_hist is None:
        print(f"参考样本 {REFERENCE_FOLDER}_{REFERENCE_DATA} 读取失败，跳过。")
        continue

    # 2. 遍历所有文件夹，构建直方图
    histograms = {}
    count_stats = []
    for folder in days:
        for data_subfolder in data_folders:
            folder_key = f"{folder}_{data_subfolder}"
            hist, _ = load_histogram(folder, data_subfolder, RATIO_OTHER)
            if hist is not None:
                histograms[folder_key] = hist
                # 统计细胞数量
                csv_path = os.path.join(folder, data_subfolder, "total", "merged.csv")
                df = pd.read_csv(csv_path)
                area_series = df[AREA_COL].dropna()
                total_cells = len(area_series)
                in_range_cells = ((area_series >= RANGE[0]) & (area_series <= RANGE[1])).sum()
                count_stats.append({
                    "Folder": folder_key,
                    "Total Cells": total_cells,
                    "Cells in 500-3500": in_range_cells
                })
    # 保存细胞数量统计
    count_df = pd.DataFrame(count_stats)
    count_df.to_excel(os.path.join(result_dir, "cell_count_summary.xlsx"), index=False)

    # 3. 绘制所有平滑后的直方图
    plt.figure(figsize=(16, 8))
    cmap = get_cmap("tab20", len(histograms))
    x = bin_centers
    ref_smooth = gaussian_filter1d(ref_hist, sigma=2)
    plt.plot(x, ref_smooth, label=f'{REFERENCE_FOLDER}_{REFERENCE_DATA} (ref)', color='black', linewidth=2, zorder=10)
    for idx2, (folder_key, hist) in enumerate(histograms.items()):
        smooth = gaussian_filter1d(hist, sigma=2)
        color = cmap(idx2)
        plt.plot(x, smooth, label=folder_key, color=color, alpha=0.85)
    plt.title(f"DATASET2 - Smoothed Area Distribution Histogram Curves\nRef={REFERENCE_FOLDER}_{REFERENCE_DATA} (30%)")
    plt.xlabel("Cell Area")
    plt.ylabel("Normalized Frequency")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "all_histograms_smoothed.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 计算相似度矩阵
    similarity_data = []
    for folder_key, hist in histograms.items():
        similarity_data.append({
            "Compared Folder": folder_key,
            "Histogram Intersection": compute_histogram_similarity(ref_hist, hist, 'intersection'),
            "KL Divergence": compute_histogram_similarity(ref_hist, hist, 'kl')
        })
    similarity_df = pd.DataFrame(similarity_data)
    similarity_df_inter = similarity_df.sort_values(by="Histogram Intersection", ascending=False)
    similarity_df_kl = similarity_df.sort_values(by="KL Divergence", ascending=True)
    excel_path = os.path.join(result_dir, "histogram_similarity_results.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        similarity_df_inter.to_excel(writer, sheet_name='Intersection', index=False)
        similarity_df_kl.to_excel(writer, sheet_name='KL', index=False)

    # 5. 输出结果
    print(f"\n📊 DATASET2 Similarity Analysis Results:")
    print(f"参考样本: {REFERENCE_FOLDER}_{REFERENCE_DATA}")
    print(f"总样本数: {len(histograms)}")
    print(f"分析范围: {RANGE[0]}-{RANGE[1]} 像素")

    # 输出TOP3及区分度
    print(f"\n🔝 Histogram Intersection TOP3:")
    top_inter = similarity_df_inter.head(3)
    for i, row in enumerate(top_inter.itertuples(), 1):
        print(f"{i}. {row._1}: {row._2:.4f}")
    if len(top_inter) >= 2:
        diff12 = (top_inter.iloc[0]['Histogram Intersection'] - top_inter.iloc[1]['Histogram Intersection']) / top_inter.iloc[0]['Histogram Intersection']
        print(f"区分度1-2: {(diff12*100):.2f}%")
    if len(top_inter) >= 3:
        diff23 = (top_inter.iloc[1]['Histogram Intersection'] - top_inter.iloc[2]['Histogram Intersection']) / top_inter.iloc[1]['Histogram Intersection']
        print(f"区分度2-3: {(diff23*100):.2f}%")

    print(f"\n🔝 KL Divergence TOP3:")
    top_kl = similarity_df_kl.head(3)
    for i, row in enumerate(top_kl.itertuples(), 1):
        print(f"{i}. {row._1}: {row._3:.4f}")
    if len(top_kl) >= 2:
        diff12 = (top_kl.iloc[1]['KL Divergence'] - top_kl.iloc[0]['KL Divergence']) / top_kl.iloc[0]['KL Divergence']
        print(f"区分度1-2: {(diff12*100):.2f}%")
    if len(top_kl) >= 3:
        diff23 = (top_kl.iloc[2]['KL Divergence'] - top_kl.iloc[1]['KL Divergence']) / top_kl.iloc[1]['KL Divergence']
        print(f"区分度2-3: {(diff23*100):.2f}%")

    print(f"\n📁 结果已保存到 {result_dir}/ 文件夹")
    print(f"   - cell_count_summary.xlsx: 细胞数量统计")
    print(f"   - histogram_similarity_results.xlsx: 相似度分析结果")
    print(f"   - all_histograms_smoothed.png: 直方图对比图") 