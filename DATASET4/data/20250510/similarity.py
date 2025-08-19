import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, pearsonr, entropy
from sklearn.metrics.pairwise import cosine_similarity
import os


# -------- 参数 --------
FOLDERS = ['20250510', '20250511', '20250512 6PM', '20250512 9AM', '20250513 5PM', '20250513 9AM',
           '20250514 9AM', '20250514 9PM', '20250515 9AM', '20250516 9AM', '20250517 9AM',
           '20250518 9AM', '20250519 9AM', '20250520 9AM', '20250521 9AM']
REFERENCE_FOLDER = '20250510'
AREA_COL = "area"
RATIO_OTHER = 0.5
RATIO_REF = 0.6
BINS = 50
RANGE = (500, 3500)

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
    area_data = df[AREA_COL].dropna().values
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

# 2. 遍历其他所有文件夹
histograms = {}
for folder in FOLDERS:
    hist, _ = load_histogram(folder, RATIO_OTHER)
    if hist is not None:
        histograms[folder] = hist

# 3. 绘制所有直方图
from scipy.ndimage import gaussian_filter1d

# 在绘图前对 hist 应用平滑
plt.figure(figsize=(12, 6))
plt.plot(bin_centers, gaussian_filter1d(ref_hist, sigma=2), label=f'{REFERENCE_FOLDER} (60%)', linewidth=2, color='black')

for folder, hist in histograms.items():
    smoothed_hist = gaussian_filter1d(hist, sigma=2)
    plt.plot(bin_centers, smoothed_hist, label=f'{folder} (50%)', alpha=0.6)

plt.title("Smoothed Area Distribution Histogram Curves")
plt.xlabel("Cell Area")
plt.ylabel("Normalized Frequency")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("all_histograms_smoothed.png")
plt.show()


# 4. 相似度计算与输出
print(f"\n📊 Similarity of {REFERENCE_FOLDER} vs others:\n")
for folder, hist in histograms.items():
    print(f"🔹 {REFERENCE_FOLDER} vs {folder}")
    print(f"   Cosine Similarity       : {compute_histogram_similarity(ref_hist, hist, 'cosine'):.4f}")
    print(f"   Pearson Correlation     : {compute_histogram_similarity(ref_hist, hist, 'correlation'):.4f}")
    print(f"   Histogram Intersection  : {compute_histogram_similarity(ref_hist, hist, 'intersection'):.4f}")
    print(f"   Chi-Square Distance     : {compute_histogram_similarity(ref_hist, hist, 'chi2'):.4f}")
    print(f"   KL Divergence           : {compute_histogram_similarity(ref_hist, hist, 'kl'):.4f}")
    print(f"   Wasserstein Distance    : {compute_histogram_similarity(ref_hist, hist, 'wasserstein'):.4f}\n")
