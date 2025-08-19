import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks

# 读取CSV文件，假设文件中包含一列 'area'
df = pd.read_csv('data1/total/merged.csv')

# 使用全部数据中的 'area'
areas = df['area'].values

# 计算直方图数据
counts, bin_edges = np.histogram(areas, bins=50)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# 计算每个区间的面积占比
ratios = counts / np.sum(counts)

# 使用样条插值使折线图平滑，插值目标为占比值
xnew = np.linspace(bin_centers.min(), bin_centers.max(), 300)
spline = make_interp_spline(bin_centers, ratios, k=3)
y_smooth = spline(xnew)

# 利用 find_peaks 检测平滑曲线中的峰值
peaks, _ = find_peaks(y_smooth)

plt.figure(figsize=(8, 5))

# 绘制柱状图，显示面积占比
bar_width = bin_edges[1] - bin_edges[0]
plt.bar(bin_centers, ratios, width=bar_width, color='powderblue', alpha=0.5, label='Histogram')

# 绘制平滑的折线图
plt.plot(xnew, y_smooth, linestyle='-', color='steelblue', label='Smooth Line')

# 用方块（'s'）标记所有峰值，并在峰值旁边标注横坐标和峰值
# 为避免图例重复，只对第一个峰值添加图例标签
# for i, peak in enumerate(peaks):
#     x_val = xnew[peak]
#     y_val = y_smooth[peak]
#     if i == 0:
#         plt.plot(x_val, y_val, marker='s', color='green', markersize=8, label='Peaks')
#     else:
#         plt.plot(x_val, y_val, marker='s', color='green', markersize=8)
#     # 标注显示横坐标和对应的峰值，保留1位小数和3位小数
#     plt.text(x_val, y_val, f'({x_val:.1f}, {y_val:.3f})', fontsize=9,
#              color='blue', ha='center', va='bottom')

# 找出最高峰值，并用橙色方块标记（与平滑曲线颜色一致）
if len(peaks) > 0:
    highest_peak_idx = np.argmax(y_smooth[peaks])
    highest_peak = peaks[highest_peak_idx]
    highest_peak_value = y_smooth[highest_peak]
    plt.plot(xnew[highest_peak], highest_peak_value, color='orange',
             markersize=12, label='Highest Peak')
    # 在最高峰上方再次标注横坐标和峰值（加粗显示）
    plt.text(xnew[highest_peak], highest_peak_value, f'({xnew[highest_peak]:.1f}, {highest_peak_value:.3f})',
             fontsize=10, color='orange', ha='center', va='bottom', weight='bold')

plt.xlabel('Cell Area')
plt.ylabel('Cell Area Ratio')
plt.title('Histogram & Smooth Line Plot of Cell Area Ratio')
plt.legend()

# 设置横坐标和纵坐标的固定范围（根据需要调整）
plt.xlim(0, 5000)
plt.ylim(0, 0.20)

# 保存图像
save_dir = 'data1/visualization'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'areaRatio_all.png')
plt.savefig(save_path)
plt.show()
