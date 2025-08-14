"""
藻类生长率预测模型 - 简化版本

当使用更小的面积区间间隔时，只使用原始的面积分布特征，
不添加额外的统计特征（标准差、偏度、峰度等）
"""

import pandas as pd
import os
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

SEED = 18
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ------------------- 数据准备 -------------------
# 读取温度分布数据
df_hist = pd.read_csv(r"E:\MASTER\2-spring\DATASET2\processed_data\histogram_data.csv")

# 加载生长率数据
mu_data = {
    "time": ["day2"] * 6 + ["day3"] * 6 + ["day4"] * 6 + ["day5"] * 6 + ["day6"] * 6 + ["day7"] * 6,
    "condition_value": [1, 2, 3, 4, 5, 6] * 6,
    "mu": [0.715961858, 0.742744122, 0.74893854, 0.768370602, 0.839750655, 0.771928058,
           0.737271985, 0.559615788, 0.886390786, 0.718193212, 0.726669873, 0.844045825,
           0.570857603, 0.296856449, 0.79914647, 0.616044787, 0.722030055, 0.866196787,
           0.359719086, 0.302131963, 0.453159982, 0.41685043, 0.514378025, 0.42356515,
           0.107485915, 0.284512498, 0.167756332, 0.33377318, 0.105892289, 0.017778246,
           -0.025807884, 0.363319487, -0.076189138, 0.142500063, 0.05582845, -0.136682987]
}
df_mu = pd.DataFrame(mu_data)
df = pd.merge(df_hist, df_mu, on=["time", "condition_value"])

# 简化的数据预处理函数 - 只使用原始面积分布特征
def prepare_data_simplified(df, bin_interval=30):
    """
    简化的数据准备函数，只使用原始的面积分布特征
    
    Args:
        df: 输入数据框
        bin_interval: 面积区间间隔，默认30
    """
    # 创建特征矩阵
    area_bins = np.arange(550, 3451, bin_interval)
    print(f"使用面积区间间隔: {bin_interval}")
    print(f"面积区间数量: {len(area_bins)}")
    print(f"面积区间范围: {area_bins[0]} - {area_bins[-1]}")
    
    pivot_df = df.pivot_table(
        index=["time", "condition_value"],
        columns="area_bin",
        values="frequency",
        aggfunc="first"
    )
    pivot_df = pivot_df.reindex(columns=area_bins)
    pivot_df = pivot_df.interpolate(axis=1)

    # 获取特征和标签
    X = pivot_df.values
    y = df.groupby(["time", "condition_value"])["mu"].first().values
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"标签形状: {y.shape}")
    print(f"特征数量: {X.shape[1]}")
    
    return X, y

# 测试不同的区间间隔（简化版本）
def test_different_intervals_simplified():
    """测试不同面积区间间隔对模型性能的影响（简化版本）"""
    intervals = [100, 50, 30, 20, 10]  # 测试不同的间隔
    results = {}
    
    for interval in intervals:
        print(f"\n{'='*50}")
        print(f"测试面积区间间隔: {interval} (简化版本)")
        print(f"{'='*50}")
        
        try:
            # 准备数据
            X, y = prepare_data_simplified(df, bin_interval=interval)
            
            # 数据标准化
            feature_scaler = RobustScaler()
            X_scaled = feature_scaler.fit_transform(X)
            
            # 随机打乱数据
            indices = np.random.permutation(len(X))
            X_shuffled = X_scaled[indices]
            y_shuffled = y[indices]
            
            # 划分训练集和测试集
            train_size = 20
            X_train = X_shuffled[:train_size]
            y_train = y_shuffled[:train_size]
            X_test = X_shuffled[train_size:]
            y_test = y_shuffled[train_size:]
            
            # 设置XGBoost参数
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 3,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'random_state': SEED
            }
            
            # 创建DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            # 训练模型
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, 'train'), (dtest, 'test')],
                early_stopping_rounds=10,
                verbose_eval=False
            )
            
            # 预测
            train_pred = model.predict(dtrain)
            test_pred = model.predict(dtest)
            
            # 计算评估指标
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            # 计算相对误差
            relative_errors = np.abs(y_test - test_pred) / np.abs(y_test) * 100
            mean_rel_error = np.mean(relative_errors)
            
            # 存储结果
            results[interval] = {
                'feature_count': X.shape[1],
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'mean_relative_error': mean_rel_error,
                'predictions': test_pred,
                'true_values': y_test
            }
            
            print(f"特征数量: {X.shape[1]}")
            print(f"训练集 RMSE: {train_rmse:.4f}")
            print(f"测试集 RMSE: {test_rmse:.4f}")
            print(f"训练集 R²: {train_r2:.4f}")
            print(f"测试集 R²: {test_r2:.4f}")
            print(f"平均相对误差: {mean_rel_error:.2f}%")
            
        except Exception as e:
            print(f"间隔 {interval} 处理失败: {str(e)}")
            continue
    
    return results

# 可视化比较结果
def visualize_comparison_simplified(results):
    """可视化不同区间间隔的结果比较（简化版本）"""
    intervals = list(results.keys())
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('不同面积区间间隔对模型性能的影响（简化版本）', fontsize=16)
    
    # 1. 特征数量对比
    feature_counts = [results[i]['feature_count'] for i in intervals]
    axes[0, 0].bar(intervals, feature_counts, color='skyblue')
    axes[0, 0].set_title('特征数量')
    axes[0, 0].set_xlabel('区间间隔')
    axes[0, 0].set_ylabel('特征数量')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. RMSE对比
    train_rmses = [results[i]['train_rmse'] for i in intervals]
    test_rmses = [results[i]['test_rmse'] for i in intervals]
    x_pos = np.arange(len(intervals))
    width = 0.35
    axes[0, 1].bar(x_pos - width/2, train_rmses, width, label='训练集', color='lightgreen')
    axes[0, 1].bar(x_pos + width/2, test_rmses, width, label='测试集', color='lightcoral')
    axes[0, 1].set_title('RMSE对比')
    axes[0, 1].set_xlabel('区间间隔')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(intervals)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. R²对比
    train_r2s = [results[i]['train_r2'] for i in intervals]
    test_r2s = [results[i]['test_r2'] for i in intervals]
    axes[0, 2].bar(x_pos - width/2, train_r2s, width, label='训练集', color='lightgreen')
    axes[0, 2].bar(x_pos + width/2, test_r2s, width, label='测试集', color='lightcoral')
    axes[0, 2].set_title('R²对比')
    axes[0, 2].set_xlabel('区间间隔')
    axes[0, 2].set_ylabel('R²')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(intervals)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 相对误差对比
    rel_errors = [results[i]['mean_relative_error'] for i in intervals]
    axes[1, 0].bar(intervals, rel_errors, color='gold')
    axes[1, 0].set_title('平均相对误差')
    axes[1, 0].set_xlabel('区间间隔')
    axes[1, 0].set_ylabel('相对误差 (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 预测vs真实值散点图（选择最佳间隔）
    best_interval = min(results.keys(), key=lambda x: results[x]['test_rmse'])
    best_pred = results[best_interval]['predictions']
    best_true = results[best_interval]['true_values']
    best_r2 = results[best_interval]['test_r2']
    
    axes[1, 1].scatter(best_true, best_pred, alpha=0.6, color='blue')
    axes[1, 1].plot([min(best_true), max(best_true)], [min(best_true), max(best_true)], 'r--')
    axes[1, 1].set_title(f'最佳间隔({best_interval})预测结果\nR² = {best_r2:.4f}')
    axes[1, 1].set_xlabel('真实值')
    axes[1, 1].set_ylabel('预测值')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 性能指标综合对比
    metrics = ['test_rmse', 'test_r2', 'mean_relative_error']
    metric_names = ['测试RMSE', '测试R²', '相对误差(%)']
    
    # 标准化指标以便比较
    normalized_metrics = {}
    for metric in metrics:
        values = [results[i][metric] for i in intervals]
        if metric == 'test_r2':  # R²越高越好，需要反转
            normalized_metrics[metric] = [(max(values) - v) / (max(values) - min(values)) for v in values]
        else:  # RMSE和相对误差越低越好
            normalized_metrics[metric] = [(v - min(values)) / (max(values) - min(values)) for v in values]
    
    x = np.arange(len(intervals))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        axes[1, 2].bar(x + i*width, normalized_metrics[metric], width, label=metric_names[i])
    
    axes[1, 2].set_title('标准化性能指标对比\n(越低越好)')
    axes[1, 2].set_xlabel('区间间隔')
    axes[1, 2].set_ylabel('标准化指标值')
    axes[1, 2].set_xticks(x + width)
    axes[1, 2].set_xticklabels(intervals)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('interval_comparison_simplified.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_interval

# 详细分析最佳间隔
def analyze_best_interval_simplified(results, best_interval):
    """详细分析最佳间隔的结果（简化版本）"""
    print(f"\n{'='*60}")
    print(f"最佳区间间隔分析: {best_interval} (简化版本)")
    print(f"{'='*60}")
    
    best_result = results[best_interval]
    
    print(f"特征数量: {best_result['feature_count']}")
    print(f"训练集 RMSE: {best_result['train_rmse']:.4f}")
    print(f"测试集 RMSE: {best_result['test_rmse']:.4f}")
    print(f"训练集 R²: {best_result['train_r2']:.4f}")
    print(f"测试集 R²: {best_result['test_r2']:.4f}")
    print(f"平均相对误差: {best_result['mean_relative_error']:.2f}%")
    
    # 详细预测结果
    print(f"\n详细预测结果 (间隔={best_interval}):")
    for i, (true, pred) in enumerate(zip(best_result['true_values'], best_result['predictions'])):
        abs_error = abs(true - pred)
        rel_error = abs_error / abs(true) * 100
        print(f"样本 {i+1}: 真实值 = {true:.4f}, 预测值 = {pred:.4f}, "
              f"绝对误差 = {abs_error:.4f}, 相对误差 = {rel_error:.2f}%")
    
    # 与原始间隔(100)的对比
    if 100 in results:
        original_result = results[100]
        print(f"\n与原始间隔(100)的对比:")
        print(f"RMSE改进: {original_result['test_rmse']:.4f} → {best_result['test_rmse']:.4f} "
              f"({((original_result['test_rmse'] - best_result['test_rmse']) / original_result['test_rmse'] * 100):.2f}%)")
        print(f"R²改进: {original_result['test_r2']:.4f} → {best_result['test_r2']:.4f} "
              f"({((best_result['test_r2'] - original_result['test_r2']) / original_result['test_r2'] * 100):.2f}%)")
        print(f"相对误差改进: {original_result['mean_relative_error']:.2f}% → {best_result['mean_relative_error']:.2f}% "
              f"({((original_result['mean_relative_error'] - best_result['mean_relative_error']) / original_result['mean_relative_error'] * 100):.2f}%)")

# 交叉验证（简化版本）
def cross_validate_simplified(bin_interval=30):
    """交叉验证（简化版本）"""
    print(f"\n{'='*50}")
    print(f"交叉验证 (间隔={bin_interval}, 简化版本)")
    print(f"{'='*50}")
    
    # 准备数据
    X, y = prepare_data_simplified(df, bin_interval=bin_interval)
    
    # 数据标准化
    feature_scaler = RobustScaler()
    X_scaled = feature_scaler.fit_transform(X)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []
    cv_rel_errors = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
        dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'random_state': SEED
        }
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        val_pred = model.predict(dval)
        val_rmse = np.sqrt(mean_squared_error(y_fold_val, val_pred))
        cv_scores.append(val_rmse)
        
        # 计算当前折的相对误差
        fold_rel_errors = np.abs(y_fold_val - val_pred) / np.abs(y_fold_val) * 100
        cv_rel_errors.extend(fold_rel_errors)
        
        print(f"Fold {fold+1} RMSE: {val_rmse:.4f}")
        print(f"Fold {fold+1} Mean Relative Error: {np.mean(fold_rel_errors):.2f}%")
    
    # 计算所有折的相对误差统计
    mean_cv_rel_error = np.mean(cv_rel_errors)
    std_cv_rel_error = np.std(cv_rel_errors)
    n_cv = len(cv_rel_errors)
    cv_confidence_interval = 1.96 * (std_cv_rel_error / np.sqrt(n_cv))
    
    print(f"\n交叉验证 RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"\n交叉验证相对误差统计 (%):")
    print(f"平均相对误差: {mean_cv_rel_error:.2f}%")
    print(f"95%置信区间: [{mean_cv_rel_error - cv_confidence_interval:.2f}%, {mean_cv_rel_error + cv_confidence_interval:.2f}%]")
    print(f"中位数相对误差: {np.median(cv_rel_errors):.2f}%")
    print(f"相对误差标准差: {std_cv_rel_error:.2f}%")
    print(f"最大相对误差: {np.max(cv_rel_errors):.2f}%")
    print(f"最小相对误差: {np.min(cv_rel_errors):.2f}%")

# 主执行函数
if __name__ == "__main__":
    print("开始测试不同面积区间间隔对模型性能的影响（简化版本）...")
    
    # 测试不同间隔
    results = test_different_intervals_simplified()
    
    if results:
        # 可视化比较
        best_interval = visualize_comparison_simplified(results)
        
        # 详细分析最佳间隔
        analyze_best_interval_simplified(results, best_interval)
        
        # 对最佳间隔进行交叉验证
        cross_validate_simplified(best_interval)
        
        # 保存结果到文件
        with open('interval_comparison_simplified_results.txt', 'w', encoding='utf-8') as f:
            f.write("不同面积区间间隔对模型性能的影响分析（简化版本）\n")
            f.write("="*60 + "\n\n")
            
            for interval in sorted(results.keys()):
                result = results[interval]
                f.write(f"区间间隔: {interval}\n")
                f.write(f"特征数量: {result['feature_count']}\n")
                f.write(f"训练集 RMSE: {result['train_rmse']:.4f}\n")
                f.write(f"测试集 RMSE: {result['test_rmse']:.4f}\n")
                f.write(f"训练集 R²: {result['train_r2']:.4f}\n")
                f.write(f"测试集 R²: {result['test_r2']:.4f}\n")
                f.write(f"平均相对误差: {result['mean_relative_error']:.2f}%\n")
                f.write("-"*40 + "\n")
            
            f.write(f"\n最佳区间间隔: {best_interval}\n")
        
        print(f"\n结果已保存到 'interval_comparison_simplified_results.txt'")
        print(f"可视化图表已保存到 'interval_comparison_simplified.png'")
    else:
        print("没有成功完成任何测试，请检查数据路径和格式。") 