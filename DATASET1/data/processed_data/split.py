import pandas as pd

# 读取原始CSV文件
df = pd.read_csv('histogram_data.csv')

# 按温度分组并保存
for temperature, group in df.groupby('condition_value'):
    filename = f'{temperature}.csv'
    group.to_csv(filename, index=False)
    print(f'已创建文件: {filename} 记录数: {len(group)}')

# 验证温度值范围
print('\n温度分布统计:')
print(df['condition_value'].value_counts().sort_index())