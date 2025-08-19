import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, pearsonr, entropy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import gaussian_filter1d
import os
from matplotlib.cm import get_cmap

# -------- 参数 --------
FOLDERS = ['20250510', '20250511', '20250512 6PM', '20250512 9AM', '20250513 5PM', '20250513 9AM',
           '20250514 9AM', '20250514 9PM', '20250515 9AM', '20250516 9AM', '20250517 9AM',
           '20250518 9AM', '20250519 9AM', '20250520 9AM', '20250521 9AM']
REFERENCE_FOLDER = '20250514 9AM'
AREA_COL = "area"
RATIO_OTHER = 0.5
RATIO_REF = 0.3
BINS = 60
RANGE = (500, 3500)
TOP_K = 5

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
def load_histogram(folder, ratio):
    csv_path = os.path.join(folder, "total", "merged.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping.")
        return None, None
    df = pd.read_csv(csv_path)
    
    # ✅ 仅保留面积在指定范围内的数据
    area_data = df[AREA_COL].dropna()
    area_data = area_data[(area_data >= RANGE[0]) & (area_data <= RANGE[1])].values

    # 如果数据不足则跳过
    if len(area_data) < 5:
        print(f"Warning: Too few area values in range for {folder}, skipping.")
        return None, None

    np.random.shuffle(area_data)
    sampled = area_data[:int(len(area_data) * ratio)]

    bins = np.linspace(RANGE[0], RANGE[1], BINS + 1)
    hist, _ = np.histogram(sampled, bins=bins, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return hist, bin_centers


# -------- 主执行逻辑 --------
np.random.seed(42)

# 1. 加载参考直方图（60%采样）
ref_hist, bin_centers = load_histogram(REFERENCE_FOLDER, RATIO_REF)
if ref_hist is None:
    raise RuntimeError("参考文件读取失败。")

# 2. 遍历所有文件夹，构建直方图
histograms = {}
for folder in FOLDERS:
    hist, _ = load_histogram(folder, RATIO_OTHER)
    if hist is not None:
        histograms[folder] = hist

# 2.5 统计每个文件夹中细胞总数 & 处于指定范围内的细胞数
count_stats = []

for folder in FOLDERS:
    csv_path = os.path.join(folder, "total", "merged.csv")
    if not os.path.exists(csv_path):
        count_stats.append({
            "Folder": folder,
            "Total Cells": "File Not Found",
            "Cells in 500-3500": "N/A"
        })
        continue

    df = pd.read_csv(csv_path)
    area_series = df[AREA_COL].dropna()
    total_cells = len(area_series)
    in_range_cells = ((area_series >= RANGE[0]) & (area_series <= RANGE[1])).sum()

    count_stats.append({
        "Folder": folder,
        "Total Cells": total_cells,
        "Cells in 500-3500": in_range_cells
    })

# 转为 DataFrame 并导出
count_df = pd.DataFrame(count_stats)
count_df.to_excel("results/cell_count_summary.xlsx", index=False)
print("\n📊 已保存细胞数量统计至 results/cell_count_summary.xlsx")


# 3. 绘制所有平滑后的直方图
# plt.figure(figsize=(12, 6))
# plt.plot(bin_centers, gaussian_filter1d(ref_hist, sigma=2), label=f'{REFERENCE_FOLDER} (60%)', linewidth=2, color='black')
# for folder, hist in histograms.items():
#     smoothed_hist = gaussian_filter1d(hist, sigma=2)
#     plt.plot(bin_centers, smoothed_hist, label=f'{folder} (50%)', alpha=0.6)
# plt.title("Smoothed Area Distribution Histogram Curves")
# plt.xlabel("Cell Area")
# plt.ylabel("Normalized Frequency")
# plt.legend(fontsize=8)
# plt.tight_layout()
# plt.savefig("results/all_histograms_smoothed.png")
# plt.show()

# 3️⃣ 绘制所有平滑后的直方图，增强对比度
plt.figure(figsize=(14, 7))
cmap = get_cmap("tab20", len(histograms))  # 使用高对比的tab20调色板
x = bin_centers

# 绘制参考样本曲线
ref_smooth = gaussian_filter1d(ref_hist, sigma=2)
plt.plot(x, ref_smooth, label=f'{REFERENCE_FOLDER} (ref)', color='black', linewidth=2, zorder=10)

# 绘制其他样本曲线
for idx, (folder, hist) in enumerate(histograms.items()):
    smooth = gaussian_filter1d(hist, sigma=2)
    color = cmap(idx)  # 为每条曲线分配不同颜色
    plt.plot(x, smooth, label=folder, color=color, alpha=0.85)

plt.title("Smoothed Area Distribution Histogram Curves")
plt.xlabel("Cell Area")
plt.ylabel("Normalized Frequency")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/all_histograms_smoothed.png")
plt.show()


# 4. 计算相似度矩阵
similarity_data = []
for folder, hist in histograms.items():
    similarity_data.append({
        "Compared Folder": folder,
        "Cosine Similarity": compute_histogram_similarity(ref_hist, hist, 'cosine'),
        "Pearson Correlation": compute_histogram_similarity(ref_hist, hist, 'correlation'),
        "Histogram Intersection": compute_histogram_similarity(ref_hist, hist, 'intersection'),
        "Chi-Square Distance": compute_histogram_similarity(ref_hist, hist, 'chi2'),
        "KL Divergence": compute_histogram_similarity(ref_hist, hist, 'kl'),
        "Wasserstein Distance": compute_histogram_similarity(ref_hist, hist, 'wasserstein')
    })

similarity_df = pd.DataFrame(similarity_data)
similarity_df_sorted = similarity_df.sort_values(by="Histogram Intersection", ascending=False)

# 5. 导出 Excel 表格
excel_path = "results/histogram_similarity_results.xlsx"
similarity_df_sorted.to_excel(excel_path, index=False)

# 6. 输出最相似的前 TOP_K
top_k_similar = similarity_df_sorted.head(TOP_K)


# 7. 层次聚类热力图：归一化 + 更合理的颜色映射
from sklearn.preprocessing import MinMaxScaler

# 拷贝数据
similarity_data_copy = similarity_df.copy()
folders = similarity_data_copy["Compared Folder"]

# 分组
similarity_metrics = ["Cosine Similarity", "Pearson Correlation", "Histogram Intersection"]
distance_metrics = ["Chi-Square Distance", "KL Divergence", "Wasserstein Distance"]

# 分别提取两组指标
similarity_part = similarity_data_copy[similarity_metrics]
distance_part = similarity_data_copy[distance_metrics]

# 标准化处理（0-1）使颜色强度可比
sim_scaler = MinMaxScaler()
dist_scaler = MinMaxScaler()
similarity_scaled = pd.DataFrame(sim_scaler.fit_transform(similarity_part), 
                                 columns=similarity_part.columns,
                                 index=folders)

distance_scaled = pd.DataFrame(dist_scaler.fit_transform(distance_part), 
                               columns=distance_part.columns,
                               index=folders)

# 1️⃣ 相似度指标聚类热力图
sns.clustermap(similarity_scaled,
               cmap="Reds",  # 红色表示相似度高
               annot=True,
               fmt=".2f",
               figsize=(10, 7),
               metric="euclidean",
               method="ward")

plt.suptitle(f"Similarity Clustering to {REFERENCE_FOLDER} (High=Better)", fontsize=13)
plt.savefig("results/clustering_similarity_metrics.png")
plt.show()

# 2️⃣ 差异度指标聚类热力图
sns.clustermap(distance_scaled,
               cmap="Blues_r",  # 蓝色表示差异度小更好（反色）
               annot=True,
               fmt=".2f",
               figsize=(10, 7),
               metric="euclidean",
               method="ward")

plt.suptitle(f"Distance Clustering to {REFERENCE_FOLDER} (Low=Better)", fontsize=13)
plt.savefig("results/clustering_distance_metrics.png")
plt.show()



# 输出所有对比结果（控制台）
print(f"\n📊 Similarity of {REFERENCE_FOLDER+'测试'} vs others:\n")
for folder, hist in histograms.items():
    cosine = compute_histogram_similarity(ref_hist, hist, 'cosine')
    corr = compute_histogram_similarity(ref_hist, hist, 'correlation')
    intersect = compute_histogram_similarity(ref_hist, hist, 'intersection')
    chi2 = compute_histogram_similarity(ref_hist, hist, 'chi2')
    kl = compute_histogram_similarity(ref_hist, hist, 'kl')
    wass = compute_histogram_similarity(ref_hist, hist, 'wasserstein')

    print(f"🔹 {REFERENCE_FOLDER+'测试'} vs {folder}")
    print(f"   Cosine Similarity       : {cosine:.4f}")
    print(f"   Pearson Correlation     : {corr:.4f}")
    print(f"   Histogram Intersection  : {intersect:.4f}")
    print(f"   Chi-Square Distance     : {chi2:.4f}")
    print(f"   KL Divergence           : {kl:.4f}")
    print(f"   Wasserstein Distance    : {wass:.4f}\n")

# 输出与参考样本最相似的文件夹（基于 Histogram Intersection）
best_match = similarity_df_sorted.iloc[0]
print("📌 与参考样本最相似的曲线（按 Histogram Intersection 排序）:")
print(f"🔸 文件夹: {best_match['Compared Folder']}")
print(f"   Histogram Intersection: {best_match['Histogram Intersection']:.4f}")


