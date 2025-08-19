import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# -------------------- 配置参数 --------------------
sample_ratio = 0.1               # 抽样比例（例如 0.6 表示 60%）
area_min, area_max = 500, 3500  # 面积范围限制
num_bins = 50                   # 直方图 bin 数
file_path = "merged.csv"        # 数据文件路径
output_path = "histogram_similarity.png"  # 图像输出路径

# -------------------- 数据加载与预处理 --------------------
df = pd.read_csv(file_path)
area_data = df['area'].dropna().values

# 仅保留在指定范围内的面积值
area_data = area_data[(area_data >= area_min) & (area_data <= area_max)]

# 随机划分两组样本
np.random.seed(42)
np.random.shuffle(area_data)
split_len = int(len(area_data) * sample_ratio)
area1 = area_data[:split_len]
area2 = area_data[split_len:2 * split_len]  # 防止超过范围

# -------------------- 绘制归一化直方图 --------------------
bins = np.linspace(area_min, area_max, num_bins + 1)
hist1, _ = np.histogram(area1, bins=bins, density=True)
hist2, _ = np.histogram(area2, bins=bins, density=True)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# -------------------- Wasserstein 距离 --------------------
distance = wasserstein_distance(hist1, hist2)

# -------------------- 绘图 --------------------
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, hist1, label='Random Sample 1', color='blue')
plt.plot(bin_centers, hist2, label='Random Sample 2', color='orange')
plt.xlabel('Area')
plt.ylabel('Normalized Frequency')
plt.title(f'Area Distribution Comparison\nWasserstein Distance = {distance:.6f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_path)

print(f"图像已保存为：{output_path}")
print(f"Wasserstein 距离为：{distance:.6f}")
