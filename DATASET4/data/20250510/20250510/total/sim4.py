import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import wasserstein_distance, pearsonr, entropy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d

# -------- 参数设置 --------
CSV_FILE = "merged.csv"
AREA_COL = "area"
RATIO = 0.5
BINS = 50
RANGE = (500, 3500)
NUM_ITER = 5
OUTPUT_FOLDER = "figure"

# 创建输出文件夹
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 相似度计算函数
def compute_histogram_similarity(hist1, hist2, method='cosine'):
    hist1 = np.asarray(hist1, dtype=np.float64) + 1e-10
    hist2 = np.asarray(hist2, dtype=np.float64) + 1e-10
    if method == 'cosine':
        return float(cosine_similarity([hist1], [hist2])[0][0])
    elif method == 'correlation':
        return pearsonr(hist1, hist2)[0]
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
        raise ValueError("Unsupported similarity method")

# 加载数据
df = pd.read_csv(CSV_FILE)
area_data = df[AREA_COL].dropna().values
bin_edges = np.linspace(RANGE[0], RANGE[1], BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# 多次计算
all_results = {metric: [] for metric in ['cosine', 'correlation', 'intersection', 'chi2', 'kl', 'wasserstein']}
all_histograms = []

for i in range(NUM_ITER):
    np.random.seed(42 + i)
    np.random.shuffle(area_data)
    split_idx = int(len(area_data) * RATIO)
    area1 = area_data[:split_idx]
    area2 = area_data[split_idx:]

    hist1, _ = np.histogram(area1, bins=bin_edges, density=True)
    hist2, _ = np.histogram(area2, bins=bin_edges, density=True)
    all_histograms.append((hist1, hist2))

    # 保存原始与平滑图
    # 保存原始与平滑图
    plt.figure(figsize=(10, 5))
    plt.plot(bin_centers, hist1, label='Sample A', color='blue', linestyle='--')
    plt.plot(bin_centers, hist2, label='Sample B', color='orange', linestyle='--')
    plt.plot(bin_centers, gaussian_filter1d(hist1, sigma=2), label='Sample A (smoothed)', color='blue')
    plt.plot(bin_centers, gaussian_filter1d(hist2, sigma=2), label='Sample B (smoothed)', color='orange')
    plt.title(f'Iteration {i+1} - Area Histogram Comparison')
    plt.xlabel('Area')
    plt.ylabel('Normalized Frequency')
    plt.ylim(0, 0.0012)  # 设置统一纵坐标范围
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"histogram_comparison_iter{i+1}.png"))
    plt.close()

    for method in all_results:
        score = compute_histogram_similarity(hist1, hist2, method)
        all_results[method].append(score)

# 保存每轮结果和平均值
results_df = pd.DataFrame(all_results, index=[f'Iter_{i+1}' for i in range(NUM_ITER)])
avg_results = results_df.mean().to_frame(name='Average')
combined_df = pd.concat([results_df, avg_results.T])
combined_df.to_excel("histogram_similarity_all_results.xlsx")

# 输出平均结果
combined_df.tail(1)
