import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, pearsonr, entropy
from sklearn.metrics.pairwise import cosine_similarity

# -------- 参数设置 --------
CSV_FILE = "merged.csv"           # 输入 CSV 文件
AREA_COL = "area"                 # 面积列名
RATIO = 0.5                       # 抽取比例（60%）
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
        return overlap / union  # 重叠面积占比
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

# -------- 可视化 --------
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, hist1, label='Sample A', color='blue')
plt.plot(bin_centers, hist2, label='Sample B', color='orange')
plt.xlabel('Area')
plt.ylabel('Normalized Frequency')
plt.title('Area Distribution Histogram Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("histogram_comparison.png")
plt.show()

# -------- 打印结果 --------
print("📊 Histogram Similarity Results:")
for key, val in results.items():
    print(f"{key}: {val:.6f}")
