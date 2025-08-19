import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 实验参数配置
EXPERIMENT_ROOT = "algae/5_8/20250414"  # 实验根目录路径

# 直方图参数
HIST_RANGE = (500, 3500)
NUM_BINS = 30

# 初始化 DataFrame 存储结果
hist_data = pd.DataFrame(columns=["date", "condition", "area_bin", "frequency"])

# 创建输出目录
output_dir = os.path.join(EXPERIMENT_ROOT, "processed_data")
os.makedirs(output_dir, exist_ok=True)

# 递归查找所有包含merged.csv的文件夹
def find_data_folders(root_dir):
    data_folders = []
    for root, dirs, files in os.walk(root_dir):
        if "merged.csv" in files and "total" in root:
            # 获取相对路径
            rel_path = os.path.relpath(root, root_dir)
            # 分割路径以获取日期和条件
            parts = rel_path.split(os.sep)
            if len(parts) >= 2:
                date = parts[0]
                condition = parts[1]
                data_folders.append((date, condition, root))
    return data_folders

# 获取所有数据文件夹
data_folders = find_data_folders(EXPERIMENT_ROOT)
data_folders.sort()  # 按日期和条件排序

# 处理每个数据文件夹
for date, condition, folder_path in data_folders:
    data_path = os.path.join(folder_path, "merged.csv")
    
    print(f"\n正在处理: {date}/{condition}")
    print(f"处理文件: {data_path}")
    
    # 读取数据并计算直方图
    df = pd.read_csv(data_path)
    areas = df["area"].values
    counts, bin_edges = np.histogram(areas, bins=NUM_BINS, range=HIST_RANGE)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ratios = counts / np.sum(counts)
    
    # 添加到 DataFrame
    for bin_center, ratio in zip(bin_centers, ratios):
        new_row = pd.DataFrame([{
            "date": date,
            "condition": condition,
            "area_bin": int(bin_center),
            "frequency": ratio
        }])
        hist_data = pd.concat([hist_data, new_row], ignore_index=True)
    
    # 打印基本统计信息
    print(f"\n{date}/{condition} 基本统计信息:")
    print(f"总样本数: {len(areas)}")
    print(f"面积范围: {areas.min():.2f} - {areas.max():.2f}")
    print(f"平均面积: {areas.mean():.2f}")
    print(f"中位数面积: {np.median(areas):.2f}")
    
    # 绘制单个条件的直方图
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, ratios, width=(HIST_RANGE[1]-HIST_RANGE[0])/NUM_BINS, alpha=0.7)
    plt.title(f'面积分布直方图 ({date}/{condition})')
    plt.xlabel('面积')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    
    # 保存单个条件的图片
    plt.savefig(os.path.join(output_dir, f'histogram_{date}_{condition}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 保存合并后的CSV
output_path = os.path.join(output_dir, "histogram_data.csv")
hist_data.to_csv(output_path, index=False)
print(f"\n所有数据已合并保存到: {output_path}")

# 按日期分组绘制对比图
unique_dates = hist_data['date'].unique()
for date in unique_dates:
    plt.figure(figsize=(12, 8))
    date_data = hist_data[hist_data['date'] == date]
    
    for condition in date_data['condition'].unique():
        condition_data = date_data[date_data['condition'] == condition]
        plt.plot(condition_data['area_bin'], condition_data['frequency'], 
                label=condition, alpha=0.7)
    
    plt.title(f'{date} 不同条件面积分布对比')
    plt.xlabel('面积')
    plt.ylabel('频率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'histogram_comparison_{date}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 绘制所有日期的对比图（按条件分组）
unique_conditions = hist_data['condition'].unique()
for condition in unique_conditions:
    plt.figure(figsize=(12, 8))
    condition_data = hist_data[hist_data['condition'] == condition]
    
    for date in condition_data['date'].unique():
        date_data = condition_data[condition_data['date'] == date]
        plt.plot(date_data['area_bin'], date_data['frequency'], 
                label=date, alpha=0.7)
    
    plt.title(f'条件 {condition} 不同日期面积分布对比')
    plt.xlabel('面积')
    plt.ylabel('频率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'histogram_comparison_{condition}.png'), dpi=300, bbox_inches='tight')
    plt.close()

print("所有对比图已生成完成") 