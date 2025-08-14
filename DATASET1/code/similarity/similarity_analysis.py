import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, pearsonr, entropy, chisquare
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import gaussian_filter1d
import os
from matplotlib.cm import get_cmap
import shutil
import random

# 设置中文字体以避免乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['font.size'] = 10  # 设置字体大小

# -------- 参数配置 --------
days = [f"DAY{i}" for i in range(1, 7)]
data_folders = [f"data{j}" for j in range(1, 6)]
all_refs = [(d, f) for d in days for f in data_folders]
REFERENCE_FOLDER = 'DAY3'  # 以DAY3为参考
REFERENCE_DATA = 'data1'   # 以data1为参考
AREA_COL = "area"
RATIO_OTHER = 1  # 其他样本全部数据
RATIO_REF_LIST = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # 测试不同的参考样本比例
BINS = 60
RANGE = (500, 3500)
TOP_K = 5

# 随机选取5个不同的参考样本
random.seed(42)
random_refs = random.sample(all_refs, 5)

# -------- 扩展的相似度函数 --------
def compute_histogram_similarity(hist1, hist2, method='intersection'):
    """计算两个直方图之间的相似度"""
    hist1 = np.asarray(hist1, dtype=np.float64) + 1e-10
    hist2 = np.asarray(hist2, dtype=np.float64) + 1e-10
    
    # 确保直方图归一化
    hist1_norm = hist1 / np.sum(hist1)
    hist2_norm = hist2 / np.sum(hist2)
    
    if method == 'intersection':
        return np.sum(np.minimum(hist1_norm, hist2_norm))
    elif method == 'kl':
        # KL散度 (越小越相似)
        return entropy(hist1_norm, hist2_norm)
    elif method == 'cosine':
        # 余弦相似度
        return cosine_similarity(hist1_norm.reshape(1, -1), hist2_norm.reshape(1, -1))[0, 0]
    elif method == 'pearson':
        # 皮尔逊相关系数
        corr, _ = pearsonr(hist1_norm, hist2_norm)
        return corr if not np.isnan(corr) else 0
    elif method == 'chi_square':
        # 卡方距离 (越小越相似)
        chi2 = 0.5 * np.sum((hist1_norm - hist2_norm) ** 2 / (hist1_norm + hist2_norm))
        return chi2
    elif method == 'wasserstein':
        # Wasserstein距离 (越小越相似)
        bin_centers = np.arange(len(hist1_norm))
        return wasserstein_distance(bin_centers, bin_centers, hist1_norm, hist2_norm)
    else:
        raise ValueError(f"Unsupported similarity method: {method}")

# -------- 加载与采样 --------
def load_histogram(folder, data_subfolder, ratio):
    # 修改路径指向正确的数据目录
    csv_path = os.path.join(r"D:\project\2-spring\DATASET1\data", folder, data_subfolder, "total", "merged.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping.")
        return None, None
    df = pd.read_csv(csv_path)
    area_data = df[AREA_COL].dropna()
    area_data = area_data[(area_data >= RANGE[0]) & (area_data <= RANGE[1])].values
    if len(area_data) < 5:
        print(f"Warning: Too few area values in range for {folder}/{data_subfolder}, skipping.")
        return None, None
    np.random.shuffle(area_data)
    sampled = area_data[:int(len(area_data) * ratio)]
    bins = np.linspace(RANGE[0], RANGE[1], BINS + 1)
    hist, _ = np.histogram(sampled, bins=bins, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return hist, bin_centers

# 定义所有指标
similarity_methods = {
    'Histogram Intersection': 'intersection',
    'KL Divergence': 'kl', 
    'Cosine Similarity': 'cosine',
    'Pearson Correlation': 'pearson',
    'Chi-Square Distance': 'chi_square',
    'Wasserstein Distance': 'wasserstein'
}

# 存储所有结果的字典
all_results = {}

for idx, (REFERENCE_FOLDER, REFERENCE_DATA) in enumerate(random_refs, 1):
    result_dir = os.path.join('results', f'enhanced_ref_{idx}_{REFERENCE_FOLDER}_{REFERENCE_DATA}')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    print(f"\n===== Reference Sample Group {idx}: {REFERENCE_FOLDER}_{REFERENCE_DATA} =====")
    
    # 存储当前参考样本的结果
    current_ref_results = {}
    
    # 遍历不同的参考样本比例
    for ratio_ref in RATIO_REF_LIST:
        print(f"\n--- Reference Sample Ratio: {ratio_ref:.1f} ---")
        
        np.random.seed(42)  # 确保结果可重现
        ref_hist, bin_centers = load_histogram(REFERENCE_FOLDER, REFERENCE_DATA, ratio_ref)
        if ref_hist is None:
            print(f"Failed to load reference sample {REFERENCE_FOLDER}_{REFERENCE_DATA}, skipping.")
            continue
        
        # 加载所有其他样本的直方图
        histograms = {}
        for folder in days:
            for data_subfolder in data_folders:
                folder_key = f"{folder}_{data_subfolder}"
                hist, _ = load_histogram(folder, data_subfolder, RATIO_OTHER)
                if hist is not None:
                    histograms[folder_key] = hist
        
        # 计算所有指标的相似度
        similarity_data = []
        for folder_key, hist in histograms.items():
            row = {"Compared Folder": folder_key}
            for metric_name, method_code in similarity_methods.items():
                similarity = compute_histogram_similarity(ref_hist, hist, method_code)
                row[metric_name] = similarity
            similarity_data.append(row)
        
        # 存储结果
        current_ref_results[ratio_ref] = pd.DataFrame(similarity_data)
    
    all_results[f"{REFERENCE_FOLDER}_{REFERENCE_DATA}"] = current_ref_results
    
    # 为当前参考样本创建6个指标的图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (metric_name, method_code) in enumerate(similarity_methods.items()):
        ax = axes[i]
        
        # 收集每个比例下的平均相似度
        ratios = []
        avg_similarities = []
        std_similarities = []
        
        for ratio_ref in RATIO_REF_LIST:
            if ratio_ref in current_ref_results:
                df = current_ref_results[ratio_ref]
                similarities = df[metric_name].values
                # 排除参考样本自身的比较
                ref_key = f"{REFERENCE_FOLDER}_{REFERENCE_DATA}"
                similarities = similarities[df["Compared Folder"] != ref_key]
                
                ratios.append(ratio_ref)
                avg_similarities.append(np.mean(similarities))
                std_similarities.append(np.std(similarities))
        
        # 绘制线图
        if ratios:
            ax.errorbar(ratios, avg_similarities, yerr=std_similarities, 
                       marker='o', linewidth=2, markersize=8, capsize=5)
            ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Reference Sample Ratio', fontsize=10)
            ax.set_ylabel('Similarity', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(RATIO_REF_LIST)
            
            # 根据指标类型调整y轴标签
            if method_code in ['kl', 'chi_square', 'wasserstein']:
                ax.set_ylabel('Distance (Lower is Better)', fontsize=10)
            else:
                ax.set_ylabel('Similarity (Higher is Better)', fontsize=10)
    
    plt.suptitle(f'Similarity Metrics vs Reference Sample Ratio\nReference: {REFERENCE_FOLDER}_{REFERENCE_DATA}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"similarity_vs_ratio_{REFERENCE_FOLDER}_{REFERENCE_DATA}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存详细数据到Excel
    excel_path = os.path.join(result_dir, f"detailed_similarity_results_{REFERENCE_FOLDER}_{REFERENCE_DATA}.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        for ratio_ref in RATIO_REF_LIST:
            if ratio_ref in current_ref_results:
                sheet_name = f'Ratio_{ratio_ref:.1f}'
                current_ref_results[ratio_ref].to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"📁 Results saved to {result_dir}/ folder")

# 创建汇总比较图
print(f"\n===== Creating Summary Comparison Charts =====")
summary_dir = os.path.join('results', 'summary_comparison')
if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)

# 为每个指标创建所有参考样本的比较图
for metric_name, method_code in similarity_methods.items():
    plt.figure(figsize=(12, 8))
    
    for ref_name, ref_results in all_results.items():
        ratios = []
        avg_similarities = []
        
        for ratio_ref in RATIO_REF_LIST:
            if ratio_ref in ref_results:
                df = ref_results[ratio_ref]
                similarities = df[metric_name].values
                # 排除参考样本自身的比较
                similarities = similarities[df["Compared Folder"] != ref_name]
                
                ratios.append(ratio_ref)
                avg_similarities.append(np.mean(similarities))
        
        if ratios:
            plt.plot(ratios, avg_similarities, marker='o', linewidth=2, 
                    markersize=6, label=ref_name, alpha=0.8)
    
    plt.title(f'{metric_name} - All Reference Samples Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Reference Sample Ratio', fontsize=12)
    
    if method_code in ['kl', 'chi_square', 'wasserstein']:
        plt.ylabel('Distance (Lower is Better)', fontsize=12)
    else:
        plt.ylabel('Similarity (Higher is Better)', fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(RATIO_REF_LIST)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, f"summary_{metric_name.replace(' ', '_')}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

print(f"\n🎉 All analysis completed!")
print(f"📊 Individual reference sample results saved in respective folders")
print(f"📈 Summary comparison charts saved in {summary_dir}/ folder") 