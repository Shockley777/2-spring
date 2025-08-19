import os
import glob
import pandas as pd

# 定义文件夹路径
input_folder = 'data1/features'
output_folder = 'data1/total'
output_file = os.path.join(output_folder, 'merged.csv')

# 获取所有CSV文件的路径
csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

# 读取所有CSV文件，并存储在列表中
df_list = [pd.read_csv(file) for file in csv_files]

# 合并所有DataFrame，忽略原有索引
merged_df = pd.concat(df_list, ignore_index=True)

# 新增一列表示数据序号，从1开始
merged_df.insert(0, 'sequence', range(1, len(merged_df) + 1))

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 将合并后的DataFrame保存为CSV文件，不保存行索引
merged_df.to_csv(output_file, index=False)

print(f"合并后的文件已保存至: {output_file}")
