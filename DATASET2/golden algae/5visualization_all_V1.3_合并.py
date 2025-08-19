import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 日期文件夹列表
date_folders = ['0115', '0116', '0117', '0118', '0119', '0120']

# 6 个数据文件夹名称
data_folders = ['data1', 'data2', 'data3', 'data4', 'data5', 'data6']
labels = ['22℃', '24℃', '26℃', '28℃', '30℃', 'seed']
colors = ['orange', 'deeppink', 'green', 'skyblue', 'mediumpurple', 'black']

# 统一直方图区间和平滑参数
hist_range = (500, 3500)
num_bins = 30          # 直方图的 bins 数量
smooth_points = 300    # 平滑曲线的点数

# 创建纵向子图，共享 X 轴
nrows = len(date_folders)
fig, axs = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=(10, 4.5 * nrows))  # 增加纵向高度
fig.suptitle('Cell Area Distribution at Different Dates\n(Unit: Pixel)', fontsize=14)

# 遍历日期文件夹（每个子图）
for ax, date in zip(axs, date_folders):
    # 遍历 6 个 data 文件夹
    for data_folder, color, label in zip(data_folders, colors, labels):
        csv_path = os.path.join(date, data_folder, "total", "merged.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found.")
            continue
        
        # 读取 CSV 数据
        df = pd.read_csv(csv_path)
        areas = df['area'].values

        # 计算直方图数据（统一使用相同的区间和 bin 数）
        counts, bin_edges = np.histogram(areas, bins=num_bins, range=hist_range)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # 计算归一化比例（使得总和为 1）
        ratios = counts / np.sum(counts)
        
        # 进行 B-spline 插值获得平滑曲线
        spline = make_interp_spline(bin_centers, ratios, k=3)
        x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), smooth_points)
        y_smooth = spline(x_smooth)
        
        # 绘制平滑折线图
        ax.plot(x_smooth, y_smooth, linestyle='-', color=color, label=label)
        
        # 找到平滑曲线最大值位置，并标注 `(x, label)`
        max_idx = np.argmax(y_smooth)
        max_x = x_smooth[max_idx]
        ax.text(max_x, y_smooth[max_idx], f'({max_x:.0f}, {label})', fontsize=9,
                color=color, ha='left', va='bottom')

    # 设置当前子图标题（日期）
    ax.set_title(f"Date: {date}")
    ax.set_ylabel('Density')
    ax.set_ylim(0, 0.14)  # 统一 y 轴范围

# 设置共享 X 轴标签
plt.xlabel('Cell Area (pixel)')
plt.xlim(hist_range)
plt.tight_layout(rect=[0, 0, 1, 0.93])  # 增加留白，防止标题或标签被遮挡

# 保存图像
save_dir = 'combined_visualization'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'vertical_combined_area_distribution.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 增加 dpi，避免模糊
plt.show()
