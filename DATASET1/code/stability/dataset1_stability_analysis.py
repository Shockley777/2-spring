import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.stats import entropy
from pathlib import Path
import warnings
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
warnings.filterwarnings('ignore')

# -------- 参数设置 --------
AREA_COL = "area"             # 面积列
BINS = 50                     # bin数量
RANGE = (500, 3500)           # 面积分布范围
STEP = 70                    # 每次递增样本数
EVAL_STEP = 70               # 稳定性判定评估步长（与抽样步长解耦）
MAX_SAMPLE = 50000            # 最大采样数量
THRESHOLD = 0.001             # 相似度变化阈值
MIN_SIMILARITY = 0.98      # 相似度最低要求
CONSECUTIVE = 5               # 连续几次 Δ < 阈值 判定稳定

# 推荐策略选择: "75th" (75分位数), "90th" (90分位数), "max" (最大值)
RECOMMENDATION_STRATEGY = "75th"  # 可修改为 "90th" 或 "max"

# -------- 相似度函数：Histogram Intersection --------
def compute_histogram_intersection(hist1, hist2):
    hist1 = np.asarray(hist1, dtype=np.float64) + 1e-10
    hist2 = np.asarray(hist2, dtype=np.float64) + 1e-10
    return np.sum(np.minimum(hist1, hist2)) / np.sum(np.maximum(hist1, hist2))

# -------- KL散度函数 --------
def compute_kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64) + 1e-10
    q = np.asarray(q, dtype=np.float64) + 1e-10
    return np.sum(p * np.log(p / q))

# -------- 分析单个文件的稳定性 --------
def analyze_single_file(csv_path, area_col=AREA_COL):
    """分析单个CSV文件的稳定性"""
    try:
        df = pd.read_csv(csv_path)
        if area_col not in df.columns:
            print(f"⚠️ {csv_path}: 未找到列 '{area_col}'")
            return None
            
        area_data = df[area_col].dropna().to_numpy()
        if len(area_data) < STEP * 2:
            print(f"⚠️ {csv_path}: 数据量不足 ({len(area_data)} < {STEP * 2})")
            return None
            
        area_data = np.sort(area_data)  # 排序确保稳定采样
        total_n = min(len(area_data), MAX_SAMPLE)
        
        # 计算相似度
        similarities = [1.0]
        prev_hist = None
        bins = np.linspace(RANGE[0], RANGE[1], BINS + 1)
        
        for n in range(STEP, total_n + STEP, STEP):
            current_sample = area_data[:n]
            hist, _ = np.histogram(current_sample, bins=bins, density=True)
            
            if prev_hist is not None:
                sim = compute_histogram_intersection(hist, prev_hist)
                kl = compute_kl_divergence(hist, prev_hist)
                similarities.append(sim)
                delta = sim - similarities[-2]
                print(f"Samples: {n}, Intersection Similarity: {sim:.4f}, Δ = {delta:+.4f}, KL散度: {kl:.4f}")
            prev_hist = hist
        
        # 判断稳定点（在评估网格上判定，与抽样步长解耦）
        curve_sizes = list(range(STEP * 2, STEP * (len(similarities) + 1), STEP))
        if len(curve_sizes) >= 2:
            start_eval = int(np.ceil(curve_sizes[0] / EVAL_STEP) * EVAL_STEP)
            end_eval = curve_sizes[-1]
            if start_eval <= end_eval:
                eval_sizes = list(range(start_eval, end_eval + 1, EVAL_STEP))
                eval_sims = np.interp(eval_sizes, curve_sizes, similarities[1:])
                deltas = np.abs(np.diff(eval_sims))
                stable_index = -1
                for i in range(len(deltas) - CONSECUTIVE + 1):
                    # 既要变化小，又要相似度足够高
                    if (np.all(deltas[i:i+CONSECUTIVE] < THRESHOLD) and 
                        eval_sims[i+CONSECUTIVE] >= MIN_SIMILARITY):
                        stable_index = eval_sizes[i+CONSECUTIVE]
                        print(f"   ✓ 在样本数 {stable_index} 处达到稳定 (相似度: {eval_sims[i+CONSECUTIVE]:.4f})")
                        break
            else:
                stable_index = -1
        else:
            stable_index = -1
        
        return {
            'file_path': csv_path,
            'total_cells': len(area_data),
            'stable_sample_size': stable_index,
            'similarities': similarities[1:],
            'sample_sizes': list(range(STEP * 2, STEP * (len(similarities) + 1), STEP)),
            'area_data': area_data  # 保存原始数据用于后续分析
        }
        
    except Exception as e:
        print(f"❌ {csv_path}: 处理失败 - {str(e)}")
        return None

# -------- 查找所有merged.csv文件 --------
def find_all_merged_files(base_dir="."):
    pattern = os.path.join(base_dir, "**", "total", "merged.csv")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)

# -------- 在特定样本数下的相似度插值函数 --------
def get_similarity_at_sample_size(result, target_sample_size):
    """使用线性插值估算在特定样本数下的相似度"""
    sample_sizes = result['sample_sizes']
    similarities = result['similarities']
    
    if target_sample_size <= sample_sizes[0]:
        return similarities[0]
    if target_sample_size >= sample_sizes[-1]:
        return similarities[-1]
    
    # 线性插值
    return np.interp(target_sample_size, sample_sizes, similarities)

# -------- 主分析函数 --------
def analyze_dataset1_stability():
    """分析整个DATASET1的稳定性"""
    print("🔍 开始分析DATASET1的细胞面积分布稳定性...")
    
    # 递归查找所有total/merged.csv文件
    merged_files = find_all_merged_files(".")
    print(f"📁 找到 {len(merged_files)} 个merged.csv文件")
    
    if not merged_files:
        print("❌ 未找到任何merged.csv文件")
        return
    
    # 分析每个文件
    results = []
    for file_path in merged_files:
        print(f"📊 分析: {file_path}")
        result = analyze_single_file(file_path)
        if result:
            results.append(result)
    
    if not results:
        print("❌ 没有成功分析任何文件")
        return None
    
    # 汇总结果

    print(f"\n📈 成功分析 {len(results)} 个文件")
    
    # 提取稳定样本数量
    stable_sizes = [r['stable_sample_size'] for r in results if r['stable_sample_size'] != -1]
    total_cells = [r['total_cells'] for r in results]


    # 汇总结果
    print(f"\n📈 成功分析 {len(results)} 个文件")
    
    # 提取稳定样本数量
    stable_sizes = [r['stable_sample_size'] for r in results if r['stable_sample_size'] != -1]
    total_cells = [r['total_cells'] for r in results]
    
    # 计算推荐样本数（用于后续分类）
    if stable_sizes:
        recommended_size_for_classification = int(np.percentile(stable_sizes, 75))
    else:
        recommended_size_for_classification = int(np.percentile(total_cells, 50))
    
    # 分类统计
    stable_files = [r for r in results if r['stable_sample_size'] != -1]
    unstable_files = [r for r in results if r['stable_sample_size'] == -1]
    
    # 进一步分类不稳定文件（使用DATASET1自己的推荐值）
    insufficient_samples = [r for r in unstable_files if r['total_cells'] < recommended_size_for_classification]
    poor_quality = [r for r in unstable_files if any(s != s for s in r['similarities'][:3])]  # 有nan值
    slow_converging = [r for r in unstable_files if r not in insufficient_samples and r not in poor_quality]
    
    print(f"   📊 稳定文件: {len(stable_files)} 个")
    print(f"   ⚠️  不稳定文件: {len(unstable_files)} 个")
    print(f"      - 样本量不足: {len(insufficient_samples)} 个 (< {recommended_size_for_classification})")
    print(f"      - 数据质量问题: {len(poor_quality)} 个") 
    print(f"      - 缓慢收敛: {len(slow_converging)} 个")
    
    # 统计信息
    print(f"\n📊 统计信息:")
    print(f"   - 总文件数: {len(results)}")
    print(f"   - 找到稳定点的文件数: {len(stable_sizes)}")
    print(f"   - 平均细胞总数: {np.mean(total_cells):.0f}")
    print(f"   - 中位数细胞总数: {np.median(total_cells):.0f}")
    
    if stable_sizes:
        print(f"   - 稳定样本数范围: {min(stable_sizes)} - {max(stable_sizes)}")
        print(f"   - 稳定样本数中位数: {np.median(stable_sizes):.0f}")
        print(f"   - 稳定样本数平均值: {np.mean(stable_sizes):.0f}")
        
        # 不同策略的推荐样本数量（确保是STEP的倍数）
        percentile_75_raw = np.percentile(stable_sizes, 75)
        percentile_90_raw = np.percentile(stable_sizes, 90)
        
        # 向上取整到最近的STEP倍数
        percentile_75 = int(np.ceil(percentile_75_raw / STEP) * STEP)
        percentile_90 = int(np.ceil(percentile_90_raw / STEP) * STEP)
        max_size = max(stable_sizes)  # 最大值本身就是STEP的倍数
        
        print(f"\n🎯 不同策略的推荐样本数量:")
        print(f"   📊 75分位数策略: {percentile_75} (原始值: {percentile_75_raw:.1f}, 覆盖75%文件)")
        print(f"   📊 90分位数策略: {percentile_90} (原始值: {percentile_90_raw:.1f}, 覆盖90%文件)")
        print(f"   📊 最大值策略: {max_size} (覆盖100%文件)")
        
        # 成本效益分析
        print(f"\n💰 成本效益分析:")
        print(f"   - 75分位数 → 90分位数: 增加{percentile_90-percentile_75}样本 (+{(percentile_90-percentile_75)/percentile_75*100:.1f}%), 多覆盖{90-75}%文件")
        print(f"   - 90分位数 → 最大值: 增加{max_size-percentile_90}样本 (+{(max_size-percentile_90)/percentile_90*100:.1f}%), 多覆盖{100-90}%文件")
        print(f"   - 75分位数 → 最大值: 增加{max_size-percentile_75}样本 (+{(max_size-percentile_75)/percentile_75*100:.1f}%), 多覆盖{100-75}%文件")
        
        # 根据策略选择推荐样本数
        if RECOMMENDATION_STRATEGY == "90th":
            recommended_size = percentile_90
        elif RECOMMENDATION_STRATEGY == "max":
            recommended_size = max_size
        else:  # 默认使用75分位数
            recommended_size = percentile_75
        
        print(f"\n🎯 当前采用策略: {RECOMMENDATION_STRATEGY} = {recommended_size}")
        if RECOMMENDATION_STRATEGY == "75th":
            print(f"   (平衡成本与覆盖率，如需100%覆盖可选择{max_size})")
        elif RECOMMENDATION_STRATEGY == "90th":
            print(f"   (高覆盖率策略，如需100%覆盖可选择{max_size})")
        else:
            print(f"   (100%覆盖策略，保证所有文件都稳定)")
        
        # 分析超出推荐值的文件
        above_recommended = [r for r in stable_files if r['stable_sample_size'] > recommended_size]
        if above_recommended:
            print(f"\n⚠️  超出当前推荐样本数的文件 ({len(above_recommended)}个，占{len(above_recommended)/len(stable_files)*100:.1f}%):")
            for r in above_recommended:
                file_name = os.path.basename(os.path.dirname(os.path.dirname(r['file_path'])))
                similarity_at_recommended = get_similarity_at_sample_size(r, recommended_size)
                print(f"   📁 {file_name}: 需要{r['stable_sample_size']}样本稳定, 在{recommended_size}样本时相似度={similarity_at_recommended:.4f}")
            
            if RECOMMENDATION_STRATEGY == "75th":
                print(f"\n💡 策略建议:")
                avg_similarity_at_75 = np.mean([get_similarity_at_sample_size(r, percentile_75) for r in above_recommended])
                print(f"   - 若使用75分位数({percentile_75}): 未稳定文件的平均相似度={avg_similarity_at_75:.4f}")
                if avg_similarity_at_75 >= 0.80:
                    print(f"   - ✅ 建议: 可使用75分位数，风险可控")
                else:
                    print(f"   - ⚠️  建议: 考虑使用90分位数({percentile_90})或最大值({max_size})")
        else:
            print(f"\n✅ 所有文件都在推荐样本数{recommended_size}内达到稳定")
    else:
        print("⚠️ 没有找到任何稳定点")
        median_cells = np.median(total_cells)
        # 确保推荐样本数是STEP的倍数
        recommended_size = int(np.ceil(median_cells / STEP) * STEP)
        print(f"\n🎯 推荐的代表性样本数量: {recommended_size}")
        print(f"   (基于总细胞数中位数 {median_cells:.0f} 向上取整到{STEP}的倍数)")
    
    # 保存详细结果
    save_detailed_results(results, recommended_size)
    
    # 绘制汇总图表
    plot_summary_results(results, stable_sizes, recommended_size)
    
    # 单独绘制所有文件相似度曲线
    plot_all_similarity_curves(results, recommended_size)
    
    return results, recommended_size

# -------- 保存详细结果 --------
def save_detailed_results(results, recommended_size):
    """保存详细的分析结果"""
    # 创建结果DataFrame
    data = []
    for r in results:
        # 计算500-3500范围内的有效细胞数
        if 'area_data' in r:
            effective_cells = len([x for x in r['area_data'] if RANGE[0] <= x <= RANGE[1]])
        else:
            effective_cells = None
            
        data.append({
            'file_path': r['file_path'],
            'total_cells': r['total_cells'],
            'effective_cells_500_3500': effective_cells,
            'stable_sample_size': r['stable_sample_size'] if r['stable_sample_size'] != -1 else None,
            'has_stable_point': r['stable_sample_size'] != -1
        })
    
    df_results = pd.DataFrame(data)
    
    # 保存到Excel（保存到当前脚本所在目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "dataset1_stability_analysis.xlsx")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='详细结果', index=False)
        
        # 创建汇总表
        summary_data = {
            '指标': ['总文件数', '找到稳定点文件数', '平均细胞总数', '中位数细胞总数', 
                    '稳定样本数最小值', '稳定样本数最大值', '稳定样本数中位数', '稳定样本数平均值', '推荐样本数'],
            '数值': [len(results), len([r for r in results if r['stable_sample_size'] != -1]),
                    np.mean([r['total_cells'] for r in results]),
                    np.median([r['total_cells'] for r in results]),
                    min([r['stable_sample_size'] for r in results if r['stable_sample_size'] != -1]) if any(r['stable_sample_size'] != -1 for r in results) else None,
                    max([r['stable_sample_size'] for r in results if r['stable_sample_size'] != -1]) if any(r['stable_sample_size'] != -1 for r in results) else None,
                    np.median([r['stable_sample_size'] for r in results if r['stable_sample_size'] != -1]) if any(r['stable_sample_size'] != -1 for r in results) else None,
                    np.mean([r['stable_sample_size'] for r in results if r['stable_sample_size'] != -1]) if any(r['stable_sample_size'] != -1 for r in results) else None,
                    recommended_size]
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='汇总统计', index=False)
    
    print(f"💾 详细结果已保存到: {output_file}")

# -------- 绘制汇总图表 --------
def plot_summary_results(results, stable_sizes, recommended_size):
    """绘制汇总结果图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 稳定样本数分布
    if stable_sizes:
        axes[0, 0].hist(stable_sizes, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(recommended_size, color='red', linestyle='--', linewidth=2, label=f'推荐值: {recommended_size}')
        axes[0, 0].set_title('稳定样本数分布')
        axes[0, 0].set_xlabel('稳定样本数')
        axes[0, 0].set_ylabel('文件数量')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 总细胞数分布
    total_cells = [r['total_cells'] for r in results]
    axes[0, 1].hist(total_cells, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(np.median(total_cells), color='red', linestyle='--', linewidth=2, 
                      label=f'中位数: {np.median(total_cells):.0f}')
    axes[0, 1].set_title('总细胞数分布')
    axes[0, 1].set_xlabel('总细胞数')
    axes[0, 1].set_ylabel('文件数量')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 稳定样本数 vs 总细胞数散点图
    if stable_sizes:
        stable_files = [r for r in results if r['stable_sample_size'] != -1]
        stable_cells = [r['total_cells'] for r in stable_files]
        stable_samples = [r['stable_sample_size'] for r in stable_files]
        
        axes[1, 0].scatter(stable_cells, stable_samples, alpha=0.6, color='orange')
        axes[1, 0].axhline(recommended_size, color='red', linestyle='--', linewidth=2, 
                          label=f'推荐值: {recommended_size}')
        axes[1, 0].set_title('稳定样本数 vs 总细胞数')
        axes[1, 0].set_xlabel('总细胞数')
        axes[1, 0].set_ylabel('稳定样本数')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 所有文件的相似度曲线
    if len(results) > 0:
        # 绘制所有文件的相似度曲线
        for i, result in enumerate(results):
            if len(result['similarities']) > 0:
                # 使用不同的颜色和透明度
                color = plt.cm.get_cmap('viridis')(i / len(results))
                alpha = 0.6 if i < 10 else 0.3  # 前10个文件更明显，后面的更透明
                
                axes[1, 1].plot(result['sample_sizes'], result['similarities'], 
                               marker='o', markersize=2, alpha=alpha, 
                               color=color, linewidth=1)
        
        # 添加推荐样本数的垂直线
        axes[1, 1].axvline(x=recommended_size, color='red', linestyle='--', linewidth=2, 
                          label=f'推荐样本数: {recommended_size}')
        axes[1, 1].axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
        
        axes[1, 1].set_title(f'所有文件相似度曲线 (共{len(results)}个文件)')
        axes[1, 1].set_xlabel('样本数量')
        axes[1, 1].set_ylabel('直方图交集相似度')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片到当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_file = os.path.join(script_dir, "dataset1_stability_summary.png")
    plt.savefig(image_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 汇总图表已保存到: {image_file}")

# -------- 单独绘制所有文件相似度曲线 --------
def plot_all_similarity_curves(results, recommended_size):
    """单独绘制所有文件的相似度曲线"""
    if len(results) == 0:
        print("⚠️ 没有结果数据，无法绘制相似度曲线")
        return
    
    # 创建更大的图形
    plt.figure(figsize=(15, 10))
    
    # 首先绘制所有文件作为背景（低透明度）
    for i, result in enumerate(results):
        if len(result['similarities']) > 0:
            plt.plot(result['sample_sizes'], result['similarities'], 
                    color='gray', alpha=0.15, linewidth=0.8)
    
    # 选择代表性文件进行突出显示
    stable_results = [r for r in results if r['stable_sample_size'] != -1]
    if stable_results:
        # 按稳定样本数排序
        stable_results.sort(key=lambda x: x['stable_sample_size'])
        
        # 选择代表性文件：最小、25分位、中位数、75分位、最大
        n = len(stable_results)
        if n >= 5:
            representative_indices = [0, n//4, n//2, 3*n//4, n-1]
            labels = ['最小稳定样本数', '25分位数', '中位数', '75分位数', '最大稳定样本数']
            colors = ['blue', 'green', 'orange', 'purple', 'red']
        elif n >= 3:
            representative_indices = [0, n//2, n-1]
            labels = ['最小稳定样本数', '中位数', '最大稳定样本数']
            colors = ['blue', 'orange', 'red']
        else:
            representative_indices = list(range(n))
            labels = [f'文件{i+1}' for i in range(n)]
            colors = ['blue', 'orange'][:n]
        
        # 绘制代表性曲线
        for idx, rep_idx in enumerate(representative_indices):
            result = stable_results[rep_idx]
            file_name = os.path.basename(os.path.dirname(os.path.dirname(result['file_path'])))
            
            plt.plot(result['sample_sizes'], result['similarities'], 
                    color=colors[idx], linewidth=3, alpha=0.8,
                    label=f'{labels[idx]} ({file_name})')
    
    # 如果没有稳定的文件，选择前5个文件
    else:
        for i in range(min(5, len(results))):
            result = results[i]
            if len(result['similarities']) > 0:
                file_name = os.path.basename(os.path.dirname(os.path.dirname(result['file_path'])))
                color = plt.cm.get_cmap('tab10')(i)
                
                plt.plot(result['sample_sizes'], result['similarities'], 
                        color=color, linewidth=3, alpha=0.8,
                        label=f'文件{i+1} ({file_name})')
    
    file_count = len([r for r in results if len(r['similarities']) > 0])
    
    # 添加推荐样本数的垂直线
    plt.axvline(x=recommended_size, color='red', linestyle='--', linewidth=3, 
                label=f'推荐样本数: {recommended_size}')
    
    # 添加完美相似度的水平线
    plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.7, 
                label='完美相似度 (1.0)')
    
    # 添加稳定性阈值线
    plt.axhline(y=MIN_SIMILARITY, color='orange', linestyle=':', linewidth=2, alpha=0.7, 
                label=f'最低相似度要求 ({MIN_SIMILARITY})')
    
    # 设置图形属性
    stable_count = len([r for r in results if r['stable_sample_size'] != -1])
    plt.title(f'DATASET1 相似度曲线分析\n(共 {file_count} 个文件，{stable_count} 个有稳定点)', fontsize=16, fontweight='bold')
    plt.xlabel('样本数量', fontsize=14)
    plt.ylabel('直方图交集相似度', fontsize=14)
    
    # 设置图例
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # 设置网格
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 设置坐标轴范围
    plt.ylim(0.7, 1.02)  # 聚焦在高相似度区域
    
    # 添加文本说明
    plt.text(0.02, 0.98, f'参数设置:\n• Bins: {BINS}\n• 范围: {RANGE}\n• 阈值: {THRESHOLD}\n• 最低相似度: {MIN_SIMILARITY}', 
             transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片到当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_file = os.path.join(script_dir, "dataset1_all_similarity_curves.png")
    plt.savefig(image_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 所有文件相似度曲线图已保存到: {image_file}")
    
    # 输出一些统计信息
    print(f"\n📈 相似度曲线统计:")
    print(f"   - 总文件数: {file_count}")
    print(f"   - 所有文件以灰色背景曲线显示")
    
    stable_count = len([r for r in results if r['stable_sample_size'] != -1])
    if stable_count > 0:
        print(f"   - 有稳定点的文件数: {stable_count}")
        print(f"   - 代表性文件已用彩色突出显示")
    else:
        print(f"   - 没有找到稳定点，显示前5个文件作为代表")
    
    # 分析在推荐样本数处的相似度分布
    similarities_at_recommended = []
    for result in results:
        if len(result['similarities']) > 0:
            sim_at_rec = get_similarity_at_sample_size(result, recommended_size)
            similarities_at_recommended.append(sim_at_rec)
    
    if similarities_at_recommended:
        avg_sim = np.mean(similarities_at_recommended)
        min_sim = np.min(similarities_at_recommended)
        max_sim = np.max(similarities_at_recommended)
        print(f"   - 在推荐样本数{recommended_size}处:")
        print(f"     • 平均相似度: {avg_sim:.4f}")
        print(f"     • 最低相似度: {min_sim:.4f}")
        print(f"     • 最高相似度: {max_sim:.4f}")
        below_threshold = len([s for s in similarities_at_recommended if s < MIN_SIMILARITY])
        print(f"     • 低于最低要求的文件: {below_threshold}个 ({below_threshold/len(similarities_at_recommended)*100:.1f}%)")

# -------- 主程序 --------
if __name__ == "__main__":
    print("🚀 开始DATASET1稳定性分析...")
    print("=" * 60)
    
    result_data = analyze_dataset1_stability()
    if result_data is not None:
        results, recommended_size = result_data
    else:
        results, recommended_size = [], 0
    
    print("\n" + "=" * 60)
    print("✅ 分析完成！")
    print(f"🎯 推荐用于整个DATASET1的代表性样本数量: {recommended_size}")
    
    # 显示输出文件的完整路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_file = os.path.join(script_dir, "dataset1_stability_analysis.xlsx")
    summary_image_file = os.path.join(script_dir, "dataset1_stability_summary.png")
    curves_image_file = os.path.join(script_dir, "dataset1_all_similarity_curves.png")
    
    print("📁 结果文件:")
    print(f"   - {excel_file} (详细结果)")
    print(f"   - {summary_image_file} (汇总图表)")
    print(f"   - {curves_image_file} (所有文件相似度曲线)")

