import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
from pathlib import Path
import warnings
import matplotlib
from scipy.interpolate import make_interp_spline
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
warnings.filterwarnings('ignore')

# -------- 参数设置 --------
AREA_COL = "area"             # 面积列
RANGE = (500, 3500)           # 面积分布范围
BINS = 30                     # bin数量（与参考代码一致）
STABLE_SAMPLE_SIZE = 6000     # 从稳定性分析得出的推荐样本数
N_RANDOM_SAMPLES = 3          # 进行多少次随机抽样对比
SMOOTH_POINTS = 300           # 平滑曲线的点数

def find_merged_files():
    """查找所有merged.csv文件"""
    # 获取脚本所在目录，然后回到DATASET4目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # 从comparison回到DATASET4
    pattern = os.path.join(base_dir, "data", "**", "total", "merged.csv")
    print(f"🔍 搜索路径: {pattern}")  # 调试信息
    files = glob.glob(pattern, recursive=True)
    return sorted(files)

def get_file_info(file_path):
    """从文件路径提取日期和时间信息"""
    parts = file_path.split(os.sep)
    # 找到日期部分，通常是形如 "20250510"的格式
    for part in parts:
        if len(part) >= 8 and part[:8].isdigit():
            date_part = part[:8]
            time_part = part[8:].strip() if len(part) > 8 else ""
            return date_part, time_part
    return "Unknown", ""

def select_day_file(files):
    """让用户选择要分析的某天数据"""
    print("🔍 找到以下数据文件:")
    file_info = []
    for i, file_path in enumerate(files):
        date, time = get_file_info(file_path)
        file_info.append((date, time, file_path))
        print(f"   {i+1}. {date} {time}")
    
    while True:
        try:
            choice = input(f"\n请选择要分析的文件 (1-{len(files)}, 或输入q退出): ").strip()
            if choice.lower() == 'q':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
            else:
                print("❌ 无效选择，请重新输入")
        except ValueError:
            print("❌ 请输入数字")

def load_and_filter_data(file_path):
    """加载数据并过滤面积范围"""
    try:
        df = pd.read_csv(file_path)
        if AREA_COL not in df.columns:
            print(f"❌ 文件中未找到'{AREA_COL}'列")
            return None
        
        # 过滤面积范围
        area_data = df[AREA_COL].dropna()
        filtered_data = area_data[(area_data >= RANGE[0]) & (area_data <= RANGE[1])]
        
        print(f"\n📊 数据统计:")
        print(f"   - 总细胞数: {len(area_data)}")
        print(f"   - 有效范围内细胞数 ({RANGE[0]}-{RANGE[1]}): {len(filtered_data)}")
        print(f"   - 过滤比例: {len(filtered_data)/len(area_data)*100:.1f}%")
        
        return filtered_data.values
        
    except Exception as e:
        print(f"❌ 加载数据失败: {str(e)}")
        return None

def random_sample_data(data, sample_size, n_samples=1):
    """随机抽样数据"""
    if len(data) < sample_size:
        print(f"⚠️ 数据量不足，需要{sample_size}个，实际只有{len(data)}个")
        sample_size = len(data)
    
    samples = []
    for i in range(n_samples):
        sample = np.random.choice(data, size=sample_size, replace=False)
        samples.append(sample)
    
    return samples, sample_size

def calculate_smooth_curve(data):
    """根据数据计算平滑曲线"""
    # 计算直方图数据（使用相同的区间和 bin 数）
    counts, bin_edges = np.histogram(data, bins=BINS, range=RANGE)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # 计算各区间占比（归一化）
    ratios = counts / np.sum(counts)
    
    # 进行 B-spline 插值，使曲线平滑
    if len(bin_centers) >= 4:  # 至少需要4个点才能做三次B-spline
        spline = make_interp_spline(bin_centers, ratios, k=3)  # 三次 B-spline
        x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), SMOOTH_POINTS)
        y_smooth = spline(x_smooth)
        # 确保y值非负
        y_smooth = np.maximum(y_smooth, 0)
    else:
        # 如果点数不够，使用线性插值
        x_smooth = bin_centers
        y_smooth = ratios
    
    return x_smooth, y_smooth, bin_centers, ratios

def plot_curve_comparison(full_data, sampled_data_list, actual_sample_size, file_path):
    """绘制全量数据与采样数据的曲线对比"""
    date, time = get_file_info(file_path)
    
    # 创建图形：2行2列布局
    fig = plt.figure(figsize=(16, 12))
    
    # 计算全量数据的平滑曲线
    full_x, full_y, full_centers, full_ratios = calculate_smooth_curve(full_data)
    full_mean = np.mean(full_data)
    
    # 计算采样数据的平滑曲线
    sample_curves = []
    sample_means = []
    for sampled_data in sampled_data_list:
        x, y, centers, ratios = calculate_smooth_curve(sampled_data)
        sample_curves.append((x, y, centers, ratios))
        sample_means.append(np.mean(sampled_data))
    
    # 1. 绘制全量数据曲线 (左上)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(full_x, full_y, linewidth=2, color='blue', label='全量数据分布')
    ax1.axvline(full_mean, color='red', linestyle='--', linewidth=2, 
               label=f'均值: {full_mean:.1f}')
    
    # 找到最大值点并标注
    max_idx = np.argmax(full_y)
    max_x = full_x[max_idx]
    max_y = full_y[max_idx]
    ax1.plot(max_x, max_y, 'ro', markersize=8)
    ax1.text(max_x, max_y + 0.005, f'峰值\n({max_x:.0f}, {max_y:.3f})', 
             ha='center', va='bottom', fontsize=10, color='red')
    
    ax1.set_title(f'全量数据分布曲线\n{date} {time}\n(n={len(full_data)})', 
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel('细胞面积 (Pixel)')
    ax1.set_ylabel('面积占比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(RANGE)
    
    # 2. 绘制稳定样本曲线对比 (右上)
    ax2 = plt.subplot(2, 2, 2)
    
    # 绘制全量数据作为参考
    ax2.plot(full_x, full_y, linewidth=3, color='blue', alpha=0.7, 
            label=f'全量数据 (n={len(full_data)})')
    
    # 绘制采样数据曲线
    colors = ['red', 'green', 'orange']
    for i, (x, y, centers, ratios) in enumerate(sample_curves):
        color = colors[i % len(colors)]
        ax2.plot(x, y, linewidth=2, color=color, alpha=0.8, 
                label=f'随机样本 {i+1} (n={actual_sample_size})')
        
        # 标注峰值点
        max_idx = np.argmax(y)
        max_x_sample = x[max_idx]
        max_y_sample = y[max_idx]
        ax2.plot(max_x_sample, max_y_sample, 'o', color=color, markersize=6)
    
    ax2.axvline(full_mean, color='blue', linestyle='--', linewidth=2, alpha=0.7,
               label=f'全量均值: {full_mean:.1f}')
    
    ax2.set_title(f'稳定样本与全量数据对比\n{N_RANDOM_SAMPLES}次随机抽样', 
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('细胞面积 (Pixel)')
    ax2.set_ylabel('面积占比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(RANGE)
    
    # 3. 绘制统计指标对比 (左下)
    ax3 = plt.subplot(2, 2, 3)
    
    # 准备统计数据
    categories = ['全量数据'] + [f'样本{i+1}' for i in range(len(sampled_data_list))]
    means = [full_mean] + sample_means
    stds = [np.std(full_data)] + [np.std(sample) for sample in sampled_data_list]
    
    # 计算峰值位置
    peak_positions = [full_x[np.argmax(full_y)]]
    for x, y, _, _ in sample_curves:
        peak_positions.append(x[np.argmax(y)])
    
    x_pos = np.arange(len(categories))
    width = 0.25
    
    # 绘制三个指标对比
    bars1 = ax3.bar(x_pos - width, means, width, label='均值', alpha=0.8, color='skyblue')
    bars2 = ax3.bar(x_pos, stds, width, label='标准差', alpha=0.8, color='lightcoral')
    bars3 = ax3.bar(x_pos + width, peak_positions, width, label='峰值位置', alpha=0.8, color='lightgreen')
    
    ax3.set_xlabel('数据组')
    ax3.set_ylabel('数值')
    ax3.set_title('统计指标对比', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(categories, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 在柱状图上显示数值
    for bars, values in [(bars1, means), (bars2, stds), (bars3, peak_positions)]:
        for bar, value in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{value:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 4. 绘制误差分析 (右下)
    ax4 = plt.subplot(2, 2, 4)
    
    # 计算各种误差
    mean_errors = [abs(mean - full_mean)/full_mean*100 for mean in sample_means]
    peak_errors = [abs(pos - peak_positions[0])/peak_positions[0]*100 for pos in peak_positions[1:]]
    
    sample_labels = [f'样本{i+1}' for i in range(len(sampled_data_list))]
    x_pos = np.arange(len(sample_labels))
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, mean_errors, width, label='均值误差', alpha=0.8, color='orange')
    bars2 = ax4.bar(x_pos + width/2, peak_errors, width, label='峰值位置误差', alpha=0.8, color='purple')
    
    ax4.set_xlabel('随机样本')
    ax4.set_ylabel('相对误差 (%)')
    ax4.set_title('样本代表性分析\n(相对全量数据的误差)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(sample_labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 添加误差阈值线
    ax4.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='优秀 (<1%)')
    ax4.axhline(y=3, color='orange', linestyle='--', alpha=0.7, label='良好 (<3%)')
    ax4.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='一般 (<5%)')
    
    # 显示具体数值
    for bars, errors in [(bars1, mean_errors), (bars2, peak_errors)]:
        for bar, error in zip(bars, errors):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{error:.2f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # 保存图片
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clean_time = time.replace(' ', '_').replace(':', '')
    image_file = os.path.join(script_dir, f"curve_comparison_{date}_{clean_time}.png")
    plt.savefig(image_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 曲线对比图已保存到: {image_file}")
    
    return image_file, mean_errors, peak_errors

def print_analysis_summary(full_data, sampled_data_list, actual_sample_size, mean_errors, peak_errors):
    """打印分析总结"""
    print(f"\n" + "="*60)
    print(f"📈 曲线分析总结")
    print(f"="*60)
    
    full_mean = np.mean(full_data)
    full_std = np.std(full_data)
    
    # 计算全量数据峰值
    full_x, full_y, _, _ = calculate_smooth_curve(full_data)
    full_peak = full_x[np.argmax(full_y)]
    
    print(f"🔍 全量数据 (n={len(full_data)}):")
    print(f"   均值: {full_mean:.2f}  |  标准差: {full_std:.2f}  |  峰值位置: {full_peak:.1f}")
    print(f"   范围: {np.min(full_data):.1f} - {np.max(full_data):.1f}")
    
    print(f"\n🎯 稳定样本 (n={actual_sample_size}):")
    for i, sampled_data in enumerate(sampled_data_list):
        sample_mean = np.mean(sampled_data)
        sample_std = np.std(sampled_data)
        
        # 计算样本峰值
        sample_x, sample_y, _, _ = calculate_smooth_curve(sampled_data)
        sample_peak = sample_x[np.argmax(sample_y)]
        
        print(f"   样本{i+1} - 均值: {sample_mean:.2f} | 标准差: {sample_std:.2f} | 峰值: {sample_peak:.1f}")
        print(f"            误差: 均值±{mean_errors[i]:.2f}% | 峰值±{peak_errors[i]:.2f}%")
    
    # 总体评估
    avg_mean_error = np.mean(mean_errors)
    avg_peak_error = np.mean(peak_errors)
    max_mean_error = np.max(mean_errors)
    max_peak_error = np.max(peak_errors)
    
    print(f"\n📊 代表性评估:")
    print(f"   平均均值误差: {avg_mean_error:.2f}%  |  最大均值误差: {max_mean_error:.2f}%")
    print(f"   平均峰值误差: {avg_peak_error:.2f}%  |  最大峰值误差: {max_peak_error:.2f}%")
    
    # 综合评级
    overall_error = (avg_mean_error + avg_peak_error) / 2
    if overall_error < 1:
        print(f"   ✅ 评级: 优秀 - 稳定样本数({actual_sample_size})能很好代表全量数据分布")
    elif overall_error < 3:
        print(f"   ✅ 评级: 良好 - 稳定样本数({actual_sample_size})基本能代表全量数据分布")
    elif overall_error < 5:
        print(f"   ⚠️  评级: 一般 - 稳定样本数({actual_sample_size})勉强能代表全量数据分布")
    else:
        print(f"   ❌ 评级: 需改进 - 稳定样本数({actual_sample_size})不足以代表全量数据分布")
    
    # 样本覆盖率
    coverage = actual_sample_size / len(full_data) * 100
    print(f"\n💰 成本效益:")
    print(f"   样本覆盖率: {coverage:.1f}%")
    print(f"   数据减少: {100-coverage:.1f}%")
    print(f"   分布特征保持度: {100-overall_error:.1f}%")

def main():
    print("🔍 DATASET4 面积分布曲线对比分析")
    print("=" * 50)
    print(f"💡 目标: 验证稳定样本数 {STABLE_SAMPLE_SIZE} 的分布曲线代表性")
    print(f"📊 分析方法: 全量数据 vs {N_RANDOM_SAMPLES}次随机抽样平滑曲线对比")
    print(f"📏 面积范围: {RANGE[0]} - {RANGE[1]}")
    print(f"🎯 曲线平滑点数: {SMOOTH_POINTS}")
    
    # 查找数据文件
    files = find_merged_files()
    if not files:
        print("❌ 未找到任何merged.csv文件")
        return
    
    print(f"\n找到 {len(files)} 个数据文件")
    
    # 选择要分析的文件
    selected_file = select_day_file(files)
    if not selected_file:
        print("❌ 未选择文件，程序退出")
        return
    
    date, time = get_file_info(selected_file)
    print(f"\n📅 分析目标: {date} {time}")
    print(f"📁 文件路径: {selected_file}")
    
    # 加载数据
    full_data = load_and_filter_data(selected_file)
    if full_data is None:
        return
    
    # 随机抽样
    print(f"\n🎲 进行 {N_RANDOM_SAMPLES} 次随机抽样...")
    sampled_data_list, actual_sample_size = random_sample_data(full_data, STABLE_SAMPLE_SIZE, N_RANDOM_SAMPLES)
    
    # 绘制曲线对比图
    print(f"\n📊 绘制分布曲线对比...")
    image_file, mean_errors, peak_errors = plot_curve_comparison(full_data, sampled_data_list, actual_sample_size, selected_file)
    
    # 分析总结
    print_analysis_summary(full_data, sampled_data_list, actual_sample_size, mean_errors, peak_errors)
    
    print(f"\n" + "="*60)
    print(f"✅ 分析完成！")
    print(f"📊 曲线对比图表: {os.path.basename(image_file)}")
    print(f"💡 结论: 通过曲线对比验证稳定样本数 {actual_sample_size} 的分布代表性")

if __name__ == "__main__":
    main() 