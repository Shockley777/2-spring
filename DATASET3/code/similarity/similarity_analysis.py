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
import glob

# -------- 参数配置 --------
# DATASET3的日期文件夹结构
date_folders = ['20250321', '20250410', '20250414', '20250421']
AREA_COL = "area"
RATIO_OTHER = 0.5
BINS = 60
RANGE = (500, 3500)
TOP_K = 5

# 获取所有可用的子文件夹组合
all_refs = []
for date_folder in date_folders:
    if os.path.exists(date_folder):
        # 查找该日期下的所有子文件夹
        subfolders = []
        for item in os.listdir(date_folder):
            item_path = os.path.join(date_folder, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                subfolders.append(item)
        
        # 为每个子文件夹创建参考组合
        for subfolder in subfolders:
            all_refs.append((date_folder, subfolder))

# 如果参考样本太少，使用所有可用的
if len(all_refs) < 5:
    print(f"Warning: Only {len(all_refs)} reference samples available, using all of them.")
    random_refs = all_refs
else:
    # 随机选取5个不同的参考样本
    random.seed(42)
    random_refs = random.sample(all_refs, 5)

for idx, (REFERENCE_FOLDER, REFERENCE_DATA) in enumerate(random_refs, 1):
    RATIO_REF = 0.2
    result_dir = os.path.join('results', f'random_ref_{idx}_{REFERENCE_FOLDER}_{REFERENCE_DATA}')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    print(f"\n===== 第{idx}组参考样本: {REFERENCE_FOLDER}_{REFERENCE_DATA} (20%数据) =====")

    # -------- 相似度函数 --------
    def compute_histogram_similarity(hist1, hist2, method='cosine'):
        hist1 = np.asarray(hist1, dtype=np.float64) + 1e-10
        hist2 = np.asarray(hist2, dtype=np.float64) + 1e-10
        if method == 'cosine':
            return float(cosine_similarity([hist1], [hist2])[0][0])
        elif method == 'correlation':
            return pearsonr(hist1, hist2)[0]
        elif method == 'intersection':
            return np.sum(np.minimum(hist1, hist2)) / np.sum(np.maximum(hist1, hist2))
        elif method == 'chi2':
            return 0.5 * np.sum((hist1 - hist2) ** 2 / (hist1 + hist2))
        elif method == 'kl':
            return entropy(hist1, hist2)
        elif method == 'wasserstein':
            return wasserstein_distance(hist1, hist2)
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
    
    for date_folder in date_folders:
        if os.path.exists(date_folder):
            # 查找该日期下的所有子文件夹
            for item in os.listdir(date_folder):
                item_path = os.path.join(date_folder, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    folder_key = f"{date_folder}_{item}"
                    hist, _ = load_histogram(date_folder, item, RATIO_OTHER)
                    if hist is not None:
                        histograms[folder_key] = hist
                        # 统计细胞数量
                        csv_path = os.path.join(date_folder, item, "total", "merged.csv")
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
    plt.title(f"DATASET3 - Smoothed Area Distribution Histogram Curves\nRef={REFERENCE_FOLDER}_{REFERENCE_DATA} (20%)")
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
            "Cosine Similarity": compute_histogram_similarity(ref_hist, hist, 'cosine'),
            "Pearson Correlation": compute_histogram_similarity(ref_hist, hist, 'correlation'),
            "Histogram Intersection": compute_histogram_similarity(ref_hist, hist, 'intersection'),
            "Chi-Square Distance": compute_histogram_similarity(ref_hist, hist, 'chi2'),
            "KL Divergence": compute_histogram_similarity(ref_hist, hist, 'kl'),
            "Wasserstein Distance": compute_histogram_similarity(ref_hist, hist, 'wasserstein')
        })
    similarity_df = pd.DataFrame(similarity_data)
    similarity_df_sorted = similarity_df.sort_values(by="Histogram Intersection", ascending=False)
    excel_path = os.path.join(result_dir, "histogram_similarity_results.xlsx")
    similarity_df_sorted.to_excel(excel_path, index=False)

    # 5. 层次聚类热力图
    from sklearn.preprocessing import MinMaxScaler
    similarity_data_copy = similarity_df.copy()
    folders = similarity_data_copy["Compared Folder"]
    similarity_metrics = ["Cosine Similarity", "Pearson Correlation", "Histogram Intersection"]
    distance_metrics = ["Chi-Square Distance", "KL Divergence", "Wasserstein Distance"]
    sim_scaler = MinMaxScaler()
    dist_scaler = MinMaxScaler()
    similarity_scaled = pd.DataFrame(sim_scaler.fit_transform(similarity_data_copy[similarity_metrics]), columns=similarity_metrics, index=folders)
    distance_scaled = pd.DataFrame(dist_scaler.fit_transform(similarity_data_copy[distance_metrics]), columns=distance_metrics, index=folders)
    # 相似度指标聚类热力图
    sns.clustermap(similarity_scaled, cmap="Reds", annot=True, fmt=".2f", figsize=(14, 10), metric="euclidean", method="ward")
    plt.suptitle(f"DATASET3 - Similarity Clustering to {REFERENCE_FOLDER}_{REFERENCE_DATA} (20%) (High=Better)", fontsize=13)
    plt.savefig(os.path.join(result_dir, "clustering_similarity_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    # 差异度指标聚类热力图
    sns.clustermap(distance_scaled, cmap="Blues_r", annot=True, fmt=".2f", figsize=(14, 10), metric="euclidean", method="ward")
    plt.suptitle(f"DATASET3 - Distance Clustering to {REFERENCE_FOLDER}_{REFERENCE_DATA} (20%) (Low=Better)", fontsize=13)
    plt.savefig(os.path.join(result_dir, "clustering_distance_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 参考样本 {REFERENCE_FOLDER}_{REFERENCE_DATA} 结果已保存到 {result_dir}/ 文件夹")

    # 7. 输出结果
    print(f"\n📊 DATASET3 Similarity Analysis Results:")
    print(f"参考样本: {REFERENCE_FOLDER}_{REFERENCE_DATA}")
    print(f"总样本数: {len(histograms)}")
    print(f"分析范围: {RANGE[0]}-{RANGE[1]} 像素")

    print(f"\n🔝 与参考样本最相似的 TOP {TOP_K} 样本:")
    top_k_similar = similarity_df_sorted.head(TOP_K)
    for idx, row in top_k_similar.iterrows():
        print(f"{row['Compared Folder']}: {row['Histogram Intersection']:.4f}")

    print(f"\n📁 结果已保存到 {result_dir}/ 文件夹")
    print(f"   - cell_count_summary.xlsx: 细胞数量统计")
    print(f"   - histogram_similarity_results.xlsx: 相似度分析结果")
    print(f"   - all_histograms_smoothed.png: 直方图对比图")
    print(f"   - clustering_similarity_metrics.png: 相似度聚类热力图")
    print(f"   - clustering_distance_metrics.png: 差异度聚类热力图")