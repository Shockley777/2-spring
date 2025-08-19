import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# -------- 参数设置 --------
CSV_FILE = "merged.csv"       # 输入CSV路径
AREA_COL = "area"             # 面积列
BINS = 50                     # bin数量
RANGE = (500, 3500)           # 面积分布范围
STEP = 500                    # 每次递增样本数
MAX_SAMPLE = 23000            # 最大采样数量
THRESHOLD = 0.01              # 相似度变化阈值
CONSECUTIVE = 3               # 连续几次 Δ < 阈值 判定稳定

# -------- 相似度函数：Histogram Intersection --------
def compute_histogram_intersection(hist1, hist2):
    hist1 = np.asarray(hist1, dtype=np.float64) + 1e-10
    hist2 = np.asarray(hist2, dtype=np.float64) + 1e-10
    return np.sum(np.minimum(hist1, hist2)) / np.sum(np.maximum(hist1, hist2))

# -------- 读取并预处理数据 --------
df = pd.read_csv(CSV_FILE)
area_data = df[AREA_COL].dropna().values
area_data = np.sort(area_data)  # 排序确保稳定采样
total_n = min(len(area_data), MAX_SAMPLE)

# -------- 主循环：逐步增加样本数 --------
similarities = [1.0]  # 初始相似度为1（自身与自身比较）
prev_hist = None
bins = np.linspace(RANGE[0], RANGE[1], BINS + 1)

for n in range(STEP, total_n + STEP, STEP):
    current_sample = area_data[:n]
    hist, _ = np.histogram(current_sample, bins=bins, density=True)

    if prev_hist is not None:
        sim = compute_histogram_intersection(hist, prev_hist)
        similarities.append(sim)
    prev_hist = hist

# -------- 判断稳定点 --------
deltas = np.abs(np.diff(similarities))
stable_index = -1
for i in range(len(deltas) - CONSECUTIVE + 1):
    if np.all(deltas[i:i+CONSECUTIVE] < THRESHOLD):
        stable_index = (i + 1) * STEP
        break

# -------- 输出稳定结果 --------
print("📊 曲线相似度变化（Histogram Intersection）:")
for i, sim in enumerate(similarities[1:], start=1):
    delta = sim - similarities[i - 1]
    print(f"Samples: {i * STEP}, Intersection Similarity: {sim:.4f}, Δ = {delta:.4f}")

if stable_index != -1:
    print(f"\n✅ 曲线在样本数为 {stable_index} 时趋于稳定（Δ连续小于 {THRESHOLD} 共 {CONSECUTIVE} 次）")
else:
    print("\n⚠️ 未找到稳定点，请考虑增大样本数或放宽阈值")

# -------- 绘图 --------
x_vals = np.arange(STEP * 2, STEP * (len(similarities) + 1), STEP)
y_vals = similarities[1:]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='blue', label="Histogram Intersection")
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
if stable_index != -1:
    plt.axvline(x=stable_index, color='green', linestyle='--', label=f'Stable at {stable_index}')
plt.title("Histogram Stability by Intersection Similarity")
plt.xlabel("Number of Sampled Cells")
plt.ylabel("Histogram Intersection Similarity")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("stability_curve.png")
plt.show()
