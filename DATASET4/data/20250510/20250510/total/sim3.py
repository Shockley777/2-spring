import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, pearsonr, entropy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# -------- 参数设置 --------
CSV_FILE = "merged.csv"           # 输入 CSV 文件
AREA_COL = "area"                 # 面积列名
RATIO = 0.5                       # 抽取比例
BINS = 50                         # 直方图 bin 数
RANGE = (500, 3500)               # 面积分布范围

# -------- 相似度计算函数 --------
def compute_histogram_similarity(hist1, hist2, method='cosine'):
    hist1 = np.asarray(hist1, dtype=np.float64) + 1e-10
    hist2 = np.asarray(hist2, dtype=np.float64) + 1e-10

    if method == 'cosine':
        return float(cosine_similarity([hist1], [hist2])[0][0])
    elif method == 'correlation':
        corr, _ = pearsonr(hist1, hist2)
        return corr
    elif method == 'intersection':
        overlap = np.sum(np.minimum(hist1, hist2))
        union = np.sum(np.maximum(hist1, hist2))
        return overlap / union
    elif method == 'chi2':
        return 0.5 * np.sum((hist1 - hist2) ** 2 / (hist1 + hist2))
    elif method == 'kl':
        return entropy(hist1, hist2)
    elif method == 'wasserstein':
        return wasserstein_distance(hist1, hist2)
    else:
        raise ValueError("不支持的相似度计算方法")

# -------- 数据加载与预处理 --------
df = pd.read_csv(CSV_FILE)
area_data = df[AREA_COL].dropna().values
np.random.seed(42)
np.random.shuffle(area_data)

split_idx = int(len(area_data) * RATIO)
area1 = area_data[:split_idx]
area2 = area_data[split_idx:]

# -------- 构建直方图 --------
bins = np.linspace(RANGE[0], RANGE[1], BINS + 1)
hist1, _ = np.histogram(area1, bins=bins, density=True)
hist2, _ = np.histogram(area2, bins=bins, density=True)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# -------- 相似度计算 --------
results = {
    "Cosine Similarity": compute_histogram_similarity(hist1, hist2, 'cosine'),
    "Pearson Correlation": compute_histogram_similarity(hist1, hist2, 'correlation'),
    "Histogram Intersection": compute_histogram_similarity(hist1, hist2, 'intersection'),
    "Chi-Square Distance": compute_histogram_similarity(hist1, hist2, 'chi2'),
    "KL Divergence": compute_histogram_similarity(hist1, hist2, 'kl'),
    "Wasserstein Distance": compute_histogram_similarity(hist1, hist2, 'wasserstein')
}

# -------- 可视化：直方图曲线 --------
from scipy.ndimage import gaussian_filter1d
# -------- 平滑直方图 --------
smooth_hist1 = gaussian_filter1d(hist1, sigma=2)
smooth_hist2 = gaussian_filter1d(hist2, sigma=2)
# -------- 可视化：平滑后的直方图曲线 --------
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, smooth_hist1, label='Sample A (smoothed)', color='blue')
plt.plot(bin_centers, smooth_hist2, label='Sample B (smoothed)', color='orange')
plt.xlabel('Area')
plt.ylabel('Normalized Frequency')
plt.title('Smoothed Area Distribution Histogram Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("histogram_comparison_smoothed.png")
plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(bin_centers, hist1, label='Sample A', color='blue')
# plt.plot(bin_centers, hist2, label='Sample B', color='orange')
# plt.xlabel('Area')
# plt.ylabel('Normalized Frequency')
# plt.title('Area Distribution Histogram Comparison')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("histogram_comparison.png")
# plt.show()

similarity_df = pd.DataFrame(results, index=["Sample A vs Sample B"])
scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(similarity_df.T), 
                         index=similarity_df.columns, 
                         columns=["Normalized Score"])


# -------- 保存结果 --------
similarity_df.to_excel("histogram_similarity_results.xlsx")

# -------- 打印 --------
print("📊 Histogram Similarity Results:")
for key, val in results.items():
    print(f"{key}: {val:.6f}")
