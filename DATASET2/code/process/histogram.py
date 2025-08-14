import os
import pandas as pd
import numpy as np

# 定义时间点与对应路径的映射（根据你的文件结构手动指定）
# DAY0 和 DAY1 是 seed，使用 data7 和 data8；其余 DAY 使用 data1-data6
day_mapping = [
    ("day2", "DAY2/data6/total/merged.csv"),
    ("day3", "DAY3/data6/total/merged.csv"),
    ("day4", "DAY4/data6/total/merged.csv"),
    ("day5", "DAY5/data6/total/merged.csv"),
    ("day6", "DAY6/data6/total/merged.csv"),
    ("day7", "DAY7/data6/total/merged.csv"),
]

# 统一直方图参数
hist_range = (500, 3500)
num_bins = 30

# 初始化 DataFrame 存储结果
hist_df = pd.DataFrame(columns=['time', 'area_bin', 'frequency'])

# 遍历每个时间点和对应路径
for time_label, file_path in day_mapping:
    # 读取 CSV 数据
    df = pd.read_csv(file_path)
    areas = df['area'].values

    # 计算直方图数据
    counts, bin_edges = np.histogram(areas, bins=num_bins, range=hist_range)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # 计算区间中点
    ratios = counts / np.sum(counts)  # 归一化频率

    # 将数据添加到 DataFrame
    for bin_center, ratio in zip(bin_centers, ratios):
        hist_df = hist_df.append({
            'time': time_label,
            'area_bin': int(bin_center),  # 取整，例如 500-1000的中点是750
            'frequency': ratio
        }, ignore_index=True)

# 保存为 CSV
os.makedirs('processed_data', exist_ok=True)
hist_df.to_csv('processed_data/histogram_data.csv', index=False)
print("histogram_data.csv 已生成！")