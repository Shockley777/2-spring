import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, pearsonr, entropy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import gaussian_filter1d
import os
from matplotlib.cm import get_cmap
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# -------- 参数配置 --------
MAIN_FOLDERS = ['20250321', '20250410', '20250414', '20250421']
# 每个主文件夹下的子文件夹
SUB_FOLDERS = {
    '20250321': ['20250322', '20250323', '20250324', '20250325', '20250326', 
                 '20250327', '20250328', '20250329', '20250330', '20250331', '20250401'],
    '20250410': ['20250411', '20250412', '20250413', '20250414'],
    '20250414': ['20250415', '20250416', '20250417', '20250418', '20250419', '20250420', '20250421'],
    '20250421': ['20250422', '20250423 5PM', '20250424 5PM', '20250425 5PM', 
                 '20250426 5PM', '20250427 5PM', '20250428 9AM']
}

AREA_COL = "area"
RATIO_OTHER = 0.5
RATIO_REF = 0.6
BINS = 60
RANGE = (500, 3500)
TOP_K = 5
N_FOLDS = 5  # 5折交叉验证

# 创建结果文件夹
if not os.path.exists("results_cv"):
    os.makedirs("results_cv")

# -------- 相似度函数 --------
def compute_histogram_similarity(hist1, hist2, method='cosine'):
    hist1 = np.asarray(hist1, dtype=np.float64) + 1e-10
    hist2 = np.asarray(hist2, dtype=np.float64) + 1e-10
    if method == 'cosine':
        return float(cosine_similarity([hist1], [hist2])[0][0])
    elif method == 'correlation':
        return pearsonr(hist1, hist2)[0]
    elif method == 'intersection':
        return np.sum(np.minimum(hist1, hist2)) / np.sum(np.maximum(hist1, hist2))
    elif method == 'chi2':
        return 0.5 * np.sum((hist1 - hist2) ** 2 / (hist1 + hist2))
    elif method == 'kl':
        return entropy(hist1, hist2)
    elif method == 'wasserstein':
        return wasserstein_distance(hist1, hist2)
    else:
        raise ValueError("Unsupported similarity method")

# -------- 加载与采样 --------
def load_histogram(main_folder, sub_folder, ratio):
    # 尝试不同的路径结构
    possible_paths = [
        os.path.join(main_folder, sub_folder, "total", "merged.csv"),
        os.path.join(main_folder, sub_folder, "A", "total", "merged.csv"),
        os.path.join(main_folder, sub_folder, "B", "total", "merged.csv"),
        os.path.join(main_folder, sub_folder, "C", "total", "merged.csv")
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        return None, None
    
    df = pd.read_csv(csv_path)
    
    # 仅保留面积在指定范围内的数据
    area_data = df[AREA_COL].dropna()
    area_data = area_data[(area_data >= RANGE[0]) & (area_data <= RANGE[1])].values

    # 如果数据不足则跳过
    if len(area_data) < 5:
        return None, None

    np.random.shuffle(area_data)
    sampled = area_data[:int(len(area_data) * ratio)]

    bins = np.linspace(RANGE[0], RANGE[1], BINS + 1)
    hist, _ = np.histogram(sampled, bins=bins, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return hist, bin_centers

# -------- 获取所有可用样本 --------
def get_all_available_samples():
    """获取所有可用的样本列表"""
    available_samples = []
    
    for main_folder in MAIN_FOLDERS:
        for sub_folder in SUB_FOLDERS[main_folder]:
            folder_key = f"{main_folder}_{sub_folder}"
            hist, _ = load_histogram(main_folder, sub_folder, RATIO_OTHER)
            if hist is not None:
                available_samples.append({
                    'folder_key': folder_key,
                    'main_folder': main_folder,
                    'sub_folder': sub_folder,
                    'histogram': hist
                })
    
    return available_samples

# -------- 单次相似度分析 --------
def run_single_analysis(reference_sample, other_samples, fold_idx):
    """运行单次相似度分析"""
    print(f"\n🔄 第 {fold_idx + 1} 折分析 - 参考样本: {reference_sample['folder_key']}")
    
    # 计算相似度
    similarity_data = []
    for sample in other_samples:
        similarity_data.append({
            "Compared Folder": sample['folder_key'],
            "Reference Folder": reference_sample['folder_key'],
            "Fold": fold_idx + 1,
            "Cosine Similarity": compute_histogram_similarity(reference_sample['histogram'], sample['histogram'], 'cosine'),
            "Pearson Correlation": compute_histogram_similarity(reference_sample['histogram'], sample['histogram'], 'correlation'),
            "Histogram Intersection": compute_histogram_similarity(reference_sample['histogram'], sample['histogram'], 'intersection'),
            "Chi-Square Distance": compute_histogram_similarity(reference_sample['histogram'], sample['histogram'], 'chi2'),
            "KL Divergence": compute_histogram_similarity(reference_sample['histogram'], sample['histogram'], 'kl'),
            "Wasserstein Distance": compute_histogram_similarity(reference_sample['histogram'], sample['histogram'], 'wasserstein')
        })
    
    return pd.DataFrame(similarity_data)

# -------- 主执行逻辑 --------
def main():
    print("🔬 DATASET3 5折交叉验证相似度分析")
    print("="*60)
    
    # 设置随机种子
    np.random.seed(42)
    
    # 获取所有可用样本
    print("📊 加载所有样本...")
    all_samples = get_all_available_samples()
    print(f"✅ 共找到 {len(all_samples)} 个有效样本")
    
    if len(all_samples) < N_FOLDS:
        print(f"❌ 样本数量 ({len(all_samples)}) 少于折数 ({N_FOLDS})，无法进行交叉验证")
        return
    
    # 5折交叉验证
    all_results = []
    fold_results = []
    
    # 随机选择5个不同的参考样本
    np.random.shuffle(all_samples)
    reference_samples = all_samples[:N_FOLDS]
    
    for fold_idx, reference_sample in enumerate(reference_samples):
        print(f"\n🔄 第 {fold_idx + 1} 折分析")
        print(f"   参考样本: {reference_sample['folder_key']}")
        
        # 其他样本作为测试集
        other_samples = [s for s in all_samples if s['folder_key'] != reference_sample['folder_key']]
        
        # 运行分析
        fold_df = run_single_analysis(reference_sample, other_samples, fold_idx)
        all_results.append(fold_df)
        
        # 统计当前折的结果
        fold_stats = {
            'fold': fold_idx + 1,
            'reference': reference_sample['folder_key'],
            'mean_intersection': fold_df['Histogram Intersection'].mean(),
            'std_intersection': fold_df['Histogram Intersection'].std(),
            'max_intersection': fold_df['Histogram Intersection'].max(),
            'min_intersection': fold_df['Histogram Intersection'].min(),
            'sample_count': len(fold_df)
        }
        fold_results.append(fold_stats)
        
        print(f"   平均相似度: {fold_stats['mean_intersection']:.4f} ± {fold_stats['std_intersection']:.4f}")
    
    # 合并所有结果
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # 保存详细结果
    combined_df.to_excel("results_cv/all_folds_similarity_results.xlsx", index=False)
    
    # 创建折间统计
    fold_summary = pd.DataFrame(fold_results)
    fold_summary.to_excel("results_cv/fold_summary.xlsx", index=False)
    
    # 计算总体统计
    print(f"\n📊 5折交叉验证总体统计:")
    print(f"   总样本数: {len(all_samples)}")
    print(f"   平均相似度: {combined_df['Histogram Intersection'].mean():.4f} ± {combined_df['Histogram Intersection'].std():.4f}")
    print(f"   相似度范围: {combined_df['Histogram Intersection'].min():.4f} - {combined_df['Histogram Intersection'].max():.4f}")
    
    # 折间比较
    print(f"\n🔄 各折结果比较:")
    for _, row in fold_summary.iterrows():
        print(f"   第{row['fold']}折 ({row['reference']}): {row['mean_intersection']:.4f} ± {row['std_intersection']:.4f}")
    
    # 找出最稳定的参考样本
    most_stable_fold = fold_summary.loc[fold_summary['std_intersection'].idxmin()]
    print(f"\n🏆 最稳定的参考样本: 第{most_stable_fold['fold']}折 ({most_stable_fold['reference']})")
    print(f"   标准差: {most_stable_fold['std_intersection']:.4f}")
    
    # 可视化：折间比较
    plt.figure(figsize=(12, 8))
    
    # 箱线图比较各折结果
    plt.subplot(2, 2, 1)
    fold_data = [combined_df[combined_df['Fold'] == i+1]['Histogram Intersection'].values 
                 for i in range(N_FOLDS)]
    plt.boxplot(fold_data, labels=[f'Fold {i+1}' for i in range(N_FOLDS)])
    plt.title('各折相似度分布比较')
    plt.ylabel('直方图交集相似度')
    plt.xticks(rotation=45)
    
    # 折间统计图
    plt.subplot(2, 2, 2)
    x_pos = np.arange(len(fold_summary))
    plt.bar(x_pos, fold_summary['mean_intersection'], 
            yerr=fold_summary['std_intersection'], capsize=5)
    plt.title('各折平均相似度')
    plt.ylabel('平均相似度')
    plt.xticks(x_pos, [f'Fold {i+1}' for i in range(N_FOLDS)], rotation=45)
    
    # 参考样本影响分析
    plt.subplot(2, 2, 3)
    plt.scatter(fold_summary['mean_intersection'], fold_summary['std_intersection'], s=100)
    for i, row in fold_summary.iterrows():
        plt.annotate(f"Fold {row['fold']}", 
                    (row['mean_intersection'], row['std_intersection']),
                    xytext=(5, 5), textcoords='offset points')
    plt.xlabel('平均相似度')
    plt.ylabel('相似度标准差')
    plt.title('参考样本稳定性分析')
    
    # 总体相似度分布
    plt.subplot(2, 2, 4)
    plt.hist(combined_df['Histogram Intersection'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(combined_df['Histogram Intersection'].mean(), color='red', linestyle='--', 
                label=f'均值: {combined_df["Histogram Intersection"].mean():.3f}')
    plt.xlabel('直方图交集相似度')
    plt.ylabel('频次')
    plt.title('总体相似度分布')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results_cv/cross_validation_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出最相似的样本（基于所有折的结果）
    print(f"\n🔝 基于5折交叉验证的最相似样本 (TOP {TOP_K}):")
    overall_similarity = combined_df.groupby('Compared Folder')['Histogram Intersection'].mean().sort_values(ascending=False)
    for i, (sample, similarity) in enumerate(overall_similarity.head(TOP_K).items()):
        print(f"   {i+1}. {sample}: {similarity:.4f}")
    
    print(f"\n📁 结果已保存到 results_cv/ 文件夹")
    print(f"   - all_folds_similarity_results.xlsx: 所有折的详细结果")
    print(f"   - fold_summary.xlsx: 各折统计摘要")
    print(f"   - cross_validation_analysis.png: 交叉验证分析图")

if __name__ == "__main__":
    main() 