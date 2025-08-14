import pandas as pd

# 读取数据
hist_data = pd.read_csv("processed_data/histogram_data.csv")
growth_rate = pd.read_csv("processed_data/growth_rate.csv")

# 合并数据
merged_data = pd.merge(
    hist_data,
    growth_rate,
    on=["time", "condition_value"],
    how="inner"
)
import numpy as np

# 按时间和条件值分组提取统计特征
features = merged_data.groupby(["time", "condition_value"]).agg(
    mean_area=("area_bin", "mean"),          # 平均面积
    median_area=("area_bin", "median"),      # 中位数
    total_area=("frequency", lambda x: np.sum(x * merged_data["area_bin"])),  # 加权总面积
    skewness=("frequency", lambda x: x.skew()),    # 分布偏度
    kurtosis=("frequency", lambda x: x.kurtosis()) # 分布峰度
).reset_index()

# 合并特征与目标变量（mu）
final_data = pd.merge(features, growth_rate, on=["time", "condition_value"])
print(final_data.head())