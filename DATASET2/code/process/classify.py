import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import os

def load_reference_data():
    """加载参考数据（历史数据）"""
    # 使用绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'processed_data', 'histogram_data.csv')
    print(f"正在读取文件: {file_path}")  # 调试信息
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    df = pd.read_csv(file_path)
    curves = {}
    for (day, cn), group in df.groupby(['time', 'condition_value']):
        area = group['area_bin'].values
        freq = group['frequency'].values
        curves[(day, cn)] = (area, freq)
    return curves

def find_best_match(new_area, new_freq, curves, top_n=3, similarity_threshold=0.5):
    """
    找到与输入曲线最匹配的历史数据
    
    参数:
    new_area: 新曲线的面积数据
    new_freq: 新曲线的频率数据
    curves: 历史数据字典
    top_n: 返回前N个最匹配的结果
    similarity_threshold: 相似度阈值，低于此值给出警告
    
    返回:
    matches: 包含前N个最匹配结果的列表，每个结果包含(day, cn, score)
    area_bins: 统一的面积区间
    """
    # 统一area_bin
    area_bins = np.sort(np.unique(np.concatenate([area for area, _ in curves.values()])))
    new_freq_interp = interp1d(new_area, new_freq, bounds_error=False, fill_value=0)(area_bins)
    
    # 计算所有相似度
    scores = []
    for key, (area, freq) in curves.items():
        freq_interp = interp1d(area, freq, bounds_error=False, fill_value=0)(area_bins)
        score = euclidean(new_freq_interp, freq_interp)
        scores.append((key, score))
    
    # 按相似度排序
    scores.sort(key=lambda x: x[1])
    matches = scores[:top_n]
    
    # 检查相似度是否足够高
    if matches[0][1] > similarity_threshold:
        print(f"警告：最匹配的相似度 ({matches[0][1]:.4f}) 低于阈值 ({similarity_threshold})")
    
    return matches, area_bins

def visualize_matches(new_area, new_freq, curves, matches, area_bins):
    """可视化匹配结果"""
    plt.figure(figsize=(12, 8))
    
    # 绘制所有历史曲线
    colors = plt.cm.tab20(np.linspace(0, 1, len(curves)))
    for idx, (key, (area, freq)) in enumerate(curves.items()):
        freq_interp = interp1d(area, freq, bounds_error=False, fill_value=0)(area_bins)
        if key == matches[0][0]:
            plt.plot(area_bins, freq_interp, color='red', linewidth=2.5, 
                    label=f"Best match: {key[0]}, CN={key[1]}")
        else:
            plt.plot(area_bins, freq_interp, color=colors[idx], alpha=0.3, linewidth=1)
    
    # 绘制新曲线
    new_freq_interp = interp1d(new_area, new_freq, bounds_error=False, fill_value=0)(area_bins)
    plt.plot(area_bins, new_freq_interp, color='black', linestyle='--', linewidth=2, 
            label='New curve')
    
    plt.xlabel('Cell Area (area_bin)')
    plt.ylabel('Frequency')
    plt.title(f'Best match: {matches[0][0][0]}, CN={matches[0][0][1]}, Distance={matches[0][1]:.4f}')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

def main():
    try:
        # 加载参考数据
        curves = load_reference_data()
        print(f"成功加载数据，共有 {len(curves)} 条曲线")
        
        # 示例：使用某条历史数据作为新数据（实际应用中替换为真实的新数据）
        test_key = list(curves.keys())[0]  # 取第一条数据作为示例
        new_area, new_freq = curves[test_key]
        print(f"使用测试数据: {test_key}")
        
        # 找到最匹配的结果
        matches, area_bins = find_best_match(new_area, new_freq, curves)
        
        # 打印结果
        print("\n匹配结果：")
        for i, (key, score) in enumerate(matches, 1):
            print(f"{i}. 天数: {key[0]}, CN比例: {key[1]}, 相似度: {score:.4f}")
        
        # 可视化
        visualize_matches(new_area, new_freq, curves, matches, area_bins)
        
        # 测试代码
        test_keys = []
        train_curves = {}
        test_curves = {}

        for key, (area, freq) in curves.items():
            test_curves[key] = (area, freq)

        for test_key, (test_area, test_freq) in test_curves.items():
            train_curves = {k: v for k, v in curves.items() if k != test_key}
            test_freq_interp = interp1d(test_area, test_freq, bounds_error=False, fill_value=0)(area_bins)
            best_score = float('inf')
            best_key = None
            for key, (area, freq) in train_curves.items():
                freq_interp = interp1d(area, freq, bounds_error=False, fill_value=0)(area_bins)
                score = euclidean(test_freq_interp, freq_interp)
                if score < best_score:
                    best_score = score
                    best_key = key

            plt.figure(figsize=(12, 8))
            colors = plt.cm.tab20(np.linspace(0, 1, len(train_curves)))
            for idx, key in enumerate(train_curves):
                area, freq = train_curves[key]
                freq_interp = interp1d(area, freq, bounds_error=False, fill_value=0)(area_bins)
                if key == best_key:
                    plt.plot(area_bins, freq_interp, color='red', linewidth=2.5, label=f"Best match: {key[0]}, CN={key[1]}")
                else:
                    plt.plot(area_bins, freq_interp, color=colors[idx], alpha=0.5, linewidth=1)
            plt.plot(area_bins, test_freq_interp, color='black', linestyle='--', linewidth=2, label='Test curve')
            plt.xlabel('Cell Area (area_bin)')
            plt.ylabel('Frequency')
            plt.title(f'Test: {test_key[0]}, CN={test_key[1]} | Best match: {best_key[0]}, CN={best_key[1]}, Distance={best_score:.4f}')
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.show()
            print(f"Test: {test_key} -> Best match: {best_key}, Distance={best_score:.4f}")
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()