import pandas as pd
import os

def merge_histogram_growth():
    # 读取数据文件
    histogram_path = 'histogram_data_combined.csv'
    growth_rate_path = 'growth_rate.csv'
    
    # 读取直方图数据
    histogram_df = pd.read_csv(histogram_path)
    
    # 读取生长率数据
    growth_df = pd.read_csv(growth_rate_path)
    
    # 处理生长率数据中的日期和条件
    growth_df['date'] = growth_df['date'].astype(str)
    growth_df['condition'] = growth_df['date'].str.extract(r'(\w+)$').fillna('')
    growth_df['date'] = growth_df['date'].str.replace(r'_\w+$', '', regex=True)
    
    # 合并数据
    merged_df = pd.merge(
        histogram_df,
        growth_df[['date', 'condition', 'growth_rate']],
        on=['date', 'condition'],
        how='left'
    )
    
    # 保存结果
    output_path = 'histogram_with_growth_rate.csv'
    merged_df.to_csv(output_path, index=False)
    print(f"数据已合并并保存到: {output_path}")

if __name__ == "__main__":
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 切换到脚本所在目录
    os.chdir(current_dir)
    # 执行合并
    merge_histogram_growth() 