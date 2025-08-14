import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.stats import entropy
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def compute_histogram_intersection(hist1, hist2):
    """计算直方图交集相似度"""
    hist1 = np.asarray(hist1, dtype=np.float64) + 1e-10
    hist2 = np.asarray(hist2, dtype=np.float64) + 1e-10
    return np.sum(np.minimum(hist1, hist2)) / np.sum(np.maximum(hist1, hist2))

def analyze_step_size_impact(csv_path, step_sizes=[100, 200, 300, 500, 1000]):
    """分析不同步长对稳定性检测的影响"""
    print(f"🔍 分析文件: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        if "area" not in df.columns:
            print(f"⚠️ 未找到area列")
            return None
            
        area_data = df["area"].dropna().to_numpy()
        area_data = np.sort(area_data)
        
        results = {}
        
        for step in step_sizes:
            print(f"\n📊 分析步长 {step}:")
            
            similarities = [1.0]
            prev_hist = None
            bins = np.linspace(500, 3500, 51)
            stable_point = -1
            
            for n in range(step, min(len(area_data), 20000), step):
                current_sample = area_data[:n]
                hist, _ = np.histogram(current_sample, bins=bins, density=True)
                
                if prev_hist is not None:
                    sim = compute_histogram_intersection(hist, prev_hist)
                    similarities.append(sim)
                    
                    # 检查稳定性（连续3次变化<0.02且相似度>0.85）
                    if len(similarities) >= 4:
                        recent_deltas = np.abs(np.diff(similarities[-4:]))
                        if np.all(recent_deltas < 0.02) and sim >= 0.85:
                            stable_point = n
                            print(f"   ✓ 稳定点: {stable_point}")
                            break
                
                prev_hist = hist
            
            results[step] = {
                'stable_point': stable_point,
                'similarities': similarities,
                'sample_sizes': list(range(step*2, step*(len(similarities)+1), step))
            }
        
        return results
        
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        return None

def plot_step_comparison(results, output_file="step_size_comparison.png"):
    """绘制不同步长的比较图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 稳定点比较
    step_sizes = []
    stable_points = []
    for step, data in results.items():
        if data['stable_point'] != -1:
            step_sizes.append(step)
            stable_points.append(data['stable_point'])
    
    ax1.bar(range(len(step_sizes)), stable_points, alpha=0.7)
    ax1.set_xticks(range(len(step_sizes)))
    ax1.set_xticklabels([f"{s}" for s in step_sizes])
    ax1.set_xlabel('步长')
    ax1.set_ylabel('稳定样本数')
    ax1.set_title('不同步长的稳定点检测结果')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(stable_points):
        ax1.text(i, v + max(stable_points)*0.01, str(v), ha='center', va='bottom')
    
    # 2. 相似度曲线比较
    colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(results)))
    for i, (step, data) in enumerate(results.items()):
        if len(data['similarities']) > 1:
            ax2.plot(data['sample_sizes'], data['similarities'], 
                    marker='o', markersize=3, alpha=0.8, color=colors[i],
                    label=f'步长 {step}')
    
    ax2.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='最低相似度要求')
    ax2.set_xlabel('样本数量')
    ax2.set_ylabel('直方图交集相似度')
    ax2.set_title('不同步长的相似度收敛曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 比较图已保存: {output_file}")

def main():
    """主分析函数"""
    print("🚀 开始步长影响验证分析...")
    print("=" * 60)
    
    # 查找一个示例文件进行分析
    pattern = os.path.join("DATASET*", "**", "total", "merged.csv")
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        print("❌ 未找到merged.csv文件")
        return
    
    # 选择第一个文件进行详细分析
    test_file = files[0]
    print(f"🎯 选择测试文件: {test_file}")
    
    # 分析不同步长的影响
    step_sizes = [100, 200, 300, 500, 1000]
    results = analyze_step_size_impact(test_file, step_sizes)
    
    if results:
        print("\n" + "=" * 60)
        print("📊 步长影响分析结果:")
        print("=" * 60)
        
        for step, data in results.items():
            stable_point = data['stable_point']
            if stable_point != -1:
                print(f"步长 {step:4d}: 稳定点 = {stable_point:4d}")
            else:
                print(f"步长 {step:4d}: 未找到稳定点")
        
        # 绘制比较图
        plot_step_comparison(results)
        
        # 分析结论
        stable_points = [data['stable_point'] for data in results.values() if data['stable_point'] != -1]
        if len(stable_points) >= 2:
            min_stable = min(stable_points)
            max_stable = max(stable_points)
            print(f"\n💡 分析结论:")
            print(f"   - 最小稳定点: {min_stable}")
            print(f"   - 最大稳定点: {max_stable}")
            print(f"   - 差异: {max_stable - min_stable} ({(max_stable - min_stable)/min_stable*100:.1f}%)")
            
            if max_stable - min_stable > min_stable * 0.3:
                print(f"   - ⚠️  步长对结果影响显著，建议使用较小步长")
            else:
                print(f"   - ✅ 步长对结果影响较小，当前设置合理")

if __name__ == "__main__":
    main() 