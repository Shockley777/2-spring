import os
import pandas as pd
import numpy as np

# 文件夹路径（包含 CSV 文件）
folder_path = '22/'  # 修改为实际路径

# 获取所有 CSV 文件列表
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.csv')]

results = []

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        # 如果文件为空或没有列，则跳过
        if df.empty or df.columns.size == 0:
            print(f"文件 {csv_file} 为空或无有效列，跳过。")
            continue
    except pd.errors.EmptyDataError:
        print(f"文件 {csv_file} 是空文件，跳过。")
        continue

    if 'area' in df.columns:
        area_data = df['area']
        mean_val = round(area_data.mean(), 2)
        std_val = round(area_data.std(), 2)
        min_val = round(area_data.min(), 2)
        max_val = round(area_data.max(), 2)
        median_val = round(area_data.median(), 2)
        count_val = area_data.count()

        # 获取不含扩展名的文件名
        filename = os.path.splitext(os.path.basename(csv_file))[0]
        
        stats = {
            'File': filename,
            'Mean': mean_val,
            'Std': std_val,
            'Min': min_val,
            'Max': max_val,
            'Median': median_val,
            'Count': count_val
        }
        results.append(stats)
    else:
        print(f"文件 {csv_file} 中不存在 'area' 列，跳过。")

results_df = pd.DataFrame(results)
output_csv = os.path.join(folder_path, '../results/area_statistics_22.csv')
results_df.to_csv(output_csv, index=False)
print(f"统计结果已保存至: {output_csv}")
