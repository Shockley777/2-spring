import os
import pandas as pd

# 实验根目录
EXPERIMENT_ROOT = "algae/5_8"

# 要处理的日期列表
dates = ["20250321", "20250410", "20250414", "20250421"]

# 初始化空的DataFrame来存储所有数据
all_data = pd.DataFrame()

# 遍历每个日期文件夹
for date in dates:
    # 构建数据文件路径
    data_path = os.path.join(EXPERIMENT_ROOT, date, "processed_data", "histogram_data.csv")
    
    # 检查文件是否存在
    if os.path.exists(data_path):
        print(f"正在读取: {data_path}")
        # 读取CSV文件
        df = pd.read_csv(data_path)
        # 添加到总DataFrame
        all_data = pd.concat([all_data, df], ignore_index=True)
    else:
        print(f"警告: 文件不存在 {data_path}")

# 创建输出目录
output_dir = os.path.join(EXPERIMENT_ROOT, "processed_data")
os.makedirs(output_dir, exist_ok=True)

# 保存合并后的数据
output_path = os.path.join(output_dir, "histogram_data.csv")
all_data.to_csv(output_path, index=False)
print(f"\n所有数据已合并保存到: {output_path}")

# 打印基本统计信息
print("\n合并后的数据统计信息:")
print(f"总行数: {len(all_data)}")
print("\n每个日期的数据量:")
print(all_data.groupby('date').size())
print("\n每个条件的数据量:")
print(all_data.groupby('condition').size()) 