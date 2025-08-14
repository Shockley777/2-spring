import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 定义 CSV 文件路径
csv_files = [
    'DAY2/data6/total/merged.csv',
    'DAY3/data6/total/merged.csv',
    'DAY4/data6/total/merged.csv',
    'DAY5/data6/total/merged.csv',
    'DAY6/data6/total/merged.csv',
    'DAY7/data6/total/merged.csv',
    'DAY0/data7/total/merged.csv',
    'DAY1/data8/total/merged.csv'
]

# 指定不同颜色和图例标签
colors = ['orange', 'deeppink', 'green', 'skyblue', 'mediumpurple','rosybrown','dimgray', 'gray']
labels = ['day2', 'day3', 'day4', 'day5', 'day6', 'day7','seed0','seed1']

plt.figure(figsize=(10, 6))

# 统一指定直方图区间
hist_range = (500, 3500)
num_bins = 30          # 直方图的 bins 数量
smooth_points = 300    # 平滑曲线的点数

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

    # 找到平滑后曲线的最大值并标注
    max_idx = np.argmax(y_smooth)
    max_x = x_smooth[max_idx]
    max_y = y_smooth[max_idx]

    # 将 label 简写为 d1、d2、…
    abbr = label.replace("day", "d")
    
    
    # 在最大值位置添加文本标注，仅显示 x 坐标和 label
    plt.text(max_x, y_smooth[max_idx], f'({max_x:.0f}, {abbr})', 
             fontsize=10, color=color, ha='left', va='bottom')

plt.xlabel('Cell Area(Pixel/Area = 24)')
plt.ylabel('Cell Area Ratio')
plt.title('Cell area distribution at Ammonium acetate (C/N 24:1) for 7 days')
plt.legend()
plt.xlim(hist_range)
plt.ylim(0, 0.16)

# 修改 x 轴刻度标签：当刻度值为 3500 时加上 " pixel"
xticks = plt.xticks()[0]
new_labels = []
for x in xticks:
    if np.isclose(x, 3500, atol=1):
        new_labels.append(f"{int(x)} pixel")
    else:
        new_labels.append(str(int(x)))
plt.xticks(xticks, new_labels)

# 保存图像
save_dir = 'combined_visualization'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'areaRatio_combined_smoothed6.png')
plt.savefig(save_path)
plt.show()
