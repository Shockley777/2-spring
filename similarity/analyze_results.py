import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_dataset_results(dataset_name, results_path):
    """分析单个数据集的结果"""
    print(f"\n{'='*60}")
    print(f"📊 {dataset_name} 相似度分析结果")
    print(f"{'='*60}")
    
    # 读取相似度结果
    similarity_file = os.path.join(results_path, "histogram_similarity_results.xlsx")
    count_file = os.path.join(results_path, "cell_count_summary.xlsx")
    
    if not os.path.exists(similarity_file):
        print(f"❌ 文件不存在: {similarity_file}")
        return None
    
    # 读取数据
    similarity_df = pd.read_excel(similarity_file)
    count_df = pd.read_excel(count_file)
    
    print(f"📈 数据概览:")
    print(f"   总样本数: {len(similarity_df)}")
    print(f"   参考样本: {similarity_df.iloc[0]['Compared Folder']}")
    
    # 相似度统计
    print(f"\n📊 相似度统计:")
    print(f"   直方图交集 - 平均值: {similarity_df['Histogram Intersection'].mean():.4f}")
    print(f"   直方图交集 - 标准差: {similarity_df['Histogram Intersection'].std():.4f}")
    print(f"   余弦相似度 - 平均值: {similarity_df['Cosine Similarity'].mean():.4f}")
    print(f"   余弦相似度 - 标准差: {similarity_df['Cosine Similarity'].std():.4f}")
    
    # 最相似的样本
    print(f"\n🔝 最相似的 TOP 5 样本:")
    top_5 = similarity_df.head(5)
    for idx, row in top_5.iterrows():
        print(f"   {row['Compared Folder']}: {row['Histogram Intersection']:.4f}")
    
    # 最不相似的样本
    print(f"\n🔻 最不相似的 TOP 5 样本:")
    bottom_5 = similarity_df.tail(5)
    for idx, row in bottom_5.iterrows():
        print(f"   {row['Compared Folder']}: {row['Histogram Intersection']:.4f}")
    
    return similarity_df, count_df

def create_comparison_plot():
    """创建三个数据集的对比图"""
    datasets = [
        ("DATASET1", "DATASET1/results"),
        ("DATASET2", "DATASET2/results"), 
        ("DATASET3", "DATASET3/results")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (dataset_name, results_path) in enumerate(datasets):
        similarity_file = os.path.join(results_path, "histogram_similarity_results.xlsx")
        if os.path.exists(similarity_file):
            df = pd.read_excel(similarity_file)
            
            # 提取时间信息
            if dataset_name == "DATASET1":
                # DAY1_data1 -> 1
                time_info = [int(folder.split('_')[0][3:]) for folder in df['Compared Folder']]
            elif dataset_name == "DATASET2":
                # DAY2_data1 -> 2
                time_info = [int(folder.split('_')[0][3:]) for folder in df['Compared Folder']]
            else:  # DATASET3
                # 20250414_20250415 -> 提取日期信息
                time_info = []
                for folder in df['Compared Folder']:
                    try:
                        # 提取子文件夹的日期
                        sub_date = folder.split('_')[1]
                        if '5PM' in sub_date:
                            day = int(sub_date.split('5PM')[0])
                        elif '9AM' in sub_date:
                            day = int(sub_date.split('9AM')[0])
                        else:
                            day = int(sub_date)
                        time_info.append(day)
                    except:
                        time_info.append(0)
            
            # 绘制散点图
            axes[idx].scatter(time_info, df['Histogram Intersection'], alpha=0.7, s=50)
            axes[idx].set_title(f'{dataset_name} - 时间 vs 相似度')
            axes[idx].set_xlabel('时间点')
            axes[idx].set_ylabel('直方图交集相似度')
            axes[idx].grid(True, alpha=0.3)
            
            # 添加趋势线
            if len(time_info) > 1:
                z = np.polyfit(time_info, df['Histogram Intersection'], 1)
                p = np.poly1d(z)
                axes[idx].plot(time_info, p(time_info), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig("dataset_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主分析函数"""
    print("🔬 微藻数据集相似度分析结果汇总")
    print("="*60)
    
    # 分析各个数据集
    results = {}
    datasets = [
        ("DATASET1", "DATASET1/results"),
        ("DATASET2", "DATASET2/results"), 
        ("DATASET3", "DATASET3/results")
    ]
    
    for dataset_name, results_path in datasets:
        result = analyze_dataset_results(dataset_name, results_path)
        if result:
            results[dataset_name] = result
    
    # 创建对比图
    print(f"\n{'='*60}")
    print("📈 创建数据集对比图...")
    create_comparison_plot()
    
    # 跨数据集比较
    print(f"\n{'='*60}")
    print("🔍 跨数据集比较分析:")
    
    dataset_stats = {}
    for dataset_name, (similarity_df, count_df) in results.items():
        dataset_stats[dataset_name] = {
            'mean_similarity': similarity_df['Histogram Intersection'].mean(),
            'std_similarity': similarity_df['Histogram Intersection'].std(),
            'max_similarity': similarity_df['Histogram Intersection'].max(),
            'min_similarity': similarity_df['Histogram Intersection'].min(),
            'sample_count': len(similarity_df)
        }
    
    print(f"\n📊 相似度统计对比:")
    for dataset_name, stats in dataset_stats.items():
        print(f"\n{dataset_name}:")
        print(f"   样本数: {stats['sample_count']}")
        print(f"   平均相似度: {stats['mean_similarity']:.4f}")
        print(f"   相似度标准差: {stats['std_similarity']:.4f}")
        print(f"   最高相似度: {stats['max_similarity']:.4f}")
        print(f"   最低相似度: {stats['min_similarity']:.4f}")
    
    # 找出最相似和最不相似的数据集
    most_similar_dataset = max(dataset_stats.items(), key=lambda x: x[1]['mean_similarity'])
    least_similar_dataset = min(dataset_stats.items(), key=lambda x: x[1]['mean_similarity'])
    
    print(f"\n🏆 数据集相似度排名:")
    print(f"   最相似的数据集: {most_similar_dataset[0]} (平均相似度: {most_similar_dataset[1]['mean_similarity']:.4f})")
    print(f"   最不相似的数据集: {least_similar_dataset[0]} (平均相似度: {least_similar_dataset[1]['mean_similarity']:.4f})")
    
    print(f"\n📁 分析完成！对比图已保存为 dataset_comparison.png")

if __name__ == "__main__":
    main() 