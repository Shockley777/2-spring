import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 定义 CSV 文件路径
csv_files = [
    'data1/total/merged.csv',
    'data2/total/merged.csv',
    'data3/total/merged.csv',
    'data4/total/merged.csv',
    'data5/total/merged.csv',
    'data6/total/merged.csv',
    'data7/total/merged.csv',
    'data8/total/merged.csv'
]

# 指定不同颜色和图例标签
colors = ['orange', 'deeppink', 'green', 'skyblue', 'mediumpurple','blue','dimgray', 'gray']
labels = ['Ammonium chloride', 'Urea', 'Ammonium acetate (C/N 32:1)', 'Ammonium acetate (C/N 48:1)', 'Ammonium acetate (C/N 40:1)','Ammonium acetate (C/N 24:1)','seed0', 'seed1']

plt.figure(figsize=(10, 6))

# 统一指定直方图区间
hist_range = (500, 3500)
num_bins = 30  # 直方图的 bins 数量
smooth_points = 300  # 平滑曲线的点数

# 遍历每个 CSV 文件
for file_path, color, label in zip(csv_files, colors, labels):
    # 读取 CSV 数据
    df = pd.read_csv(file_path)
    areas = df['area'].values

    # 计算直方图数据（使用相同的区间和 bin 数）
    counts, bin_edges = np.histogram(areas, bins=num_bins, range=hist_range)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # 计算各区间占比（归一化）
    ratios = counts / np.sum(counts)
    
    # 输出 ratios 数组、总和以及大小（检查归一化情况）
    print(f"{label} ratios:", ratios)
    print(f"Sum of ratios for {label}: {np.sum(ratios)}")
    print(f"Size of ratios for {label}: {ratios.size}")
    print('--------------')
    
    # 进行 B-spline 插值，使曲线平滑
    spline = make_interp_spline(bin_centers, ratios, k=3)  # 三次 B-spline
    x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), smooth_points)  # 生成平滑点
    y_smooth = spline(x_smooth)

    # 绘制平滑折线图
    plt.plot(x_smooth, y_smooth, linestyle='-', color=color, label=label)

    # plt.plot(bin_centers, ratios, linestyle='-', color=color, label=label)

    # 找到平滑后曲线的最大值并标注
    max_idx = np.argmax(y_smooth)
    max_x = x_smooth[max_idx]
    max_y = y_smooth[max_idx]

    # # 找到 y 轴最大值并标注
    # max_idx = np.argmax(ratios)
    # max_x = bin_centers[max_idx]
    # max_y = ratios[max_idx]

    # 添加文本标注
    # 在最大值位置添加文本标注，仅显示 x 坐标和 label
    plt.text(max_x, y_smooth[max_idx], f'({max_x:.0f}, {label})', 
             fontsize=10, color=color, ha='left', va='bottom')

plt.xlabel('Cell Area(Pixel/Area = 24)')
plt.ylabel('Cell Area Ratio')
plt.title('Cell distribution at different temperatures on Day 6')
plt.legend()
plt.xlim(hist_range)
plt.ylim(0, 0.16)

# 保存图像
save_dir = 'combined_visualization'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'areaRatio_combined_smoothed.png')
plt.savefig(save_path)
plt.show()
