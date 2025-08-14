import os
import pandas as pd
import numpy as np

# 实验参数配置
EXPERIMENT_ROOT = "."  # 替换为实验根目录路径（例如 "temperature_experiment" 或 "cn_experiment"）
CONDITION_TYPE = "cn_ratio"           # 实验类型（"temperature" 或 "cn_ratio"）
CONDITION_MAPPING = {
    "data8": 0,
    "data1": 1,    # 温度实验：data1=22℃, data2=24℃, ..., data5=30℃
    "data2": 2,
    "data3": 3,
    "data4": 4,
    "data5": 5,
    "data6": 6,
    # 如果是C/N实验，修改为：
    # "data1": 1.2, "data2": 1.5, ..., "data6": 3.5
}

# 直方图参数
HIST_RANGE = (500, 3500)
NUM_BINS = 30

# 初始化 DataFrame 存储结果
hist_data = pd.DataFrame(columns=["time", "condition_type", "condition_value", "area_bin", "frequency"])

# 遍历 DAY1-DAY6
for day in range(1,8):
    day_label = f"day{day}"
    day_dir = os.path.join(EXPERIMENT_ROOT, f"DAY{day}")  # 例如 temperature_experiment/DAY2
    
    # 跳过不存在的 DAY 目录
    if not os.path.exists(day_dir):
        print(f"跳过不存在的目录: {day_dir}")
        continue
    
    # 特殊处理种子数据（DAY0）
    if day == 0:
        seed_data_path = os.path.join(day_dir, f"data{7 if day == 0 else 8}", "total", "merged.csv")
        if not os.path.exists(seed_data_path):
            print(f"警告：种子数据路径 {seed_data_path} 不存在")
            continue
        
        # 读取数据并计算直方图
        df = pd.read_csv(seed_data_path)
        areas = df["area"].values
        counts, bin_edges = np.histogram(areas, bins=NUM_BINS, range=HIST_RANGE)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ratios = counts / np.sum(counts)
        
        # 添加到 DataFrame
        for bin_center, ratio in zip(bin_centers, ratios):
            hist_data = hist_data.append({
                "time": day_label,
                "condition_type": "seed",
                "condition_value": np.nan,
                "area_bin": int(bin_center),
                "frequency": ratio
            }, ignore_index=True)
    else:
        # 处理实验数据（遍历 data1-data5 或 data1-data6）
        for data_folder in CONDITION_MAPPING.keys():
            data_path = os.path.join(day_dir, data_folder, "total", "merged.csv")
            if not os.path.exists(data_path):
                print(f"警告：{data_path} 不存在")
                continue
            
            # 读取数据并计算直方图
            df = pd.read_csv(data_path)
            areas = df["area"].values
            counts, bin_edges = np.histogram(areas, bins=NUM_BINS, range=HIST_RANGE)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ratios = counts / np.sum(counts)
            
            # 添加到 DataFrame
            for bin_center, ratio in zip(bin_centers, ratios):
                # 替换为 pd.concat()
                new_row = pd.DataFrame([{
                    "time": day_label,
                    "condition_type": CONDITION_TYPE,
                    "condition_value": CONDITION_MAPPING[data_folder],
                    "area_bin": int(bin_center),
                    "frequency": ratio
                }])
                hist_data = pd.concat([hist_data, new_row], ignore_index=True)
# 保存为 CSV
os.makedirs("processed_data", exist_ok=True)
hist_data.to_csv("processed_data/histogram_data.csv", index=False)
print("histogram_data.csv 已生成！")