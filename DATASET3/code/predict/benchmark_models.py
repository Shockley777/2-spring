"""
多模型基准评测 - DATASET3 (藻类面积分布)

功能：
- 基于 DATASET3 的面积分布直方图数据进行建模
- 特征：面积分布直方图 + 时间 + 条件值 + 统计特征
- 统一随机划分训练/测试（总样本约100+，90%训练，10%测试）
- 比较多种模型：线性/正则化/核方法/树模型/XGBoost（可选 LightGBM、CatBoost）
- 指标：Test-RMSE、Test-MAE、Test-R²、5折CV-RMSE（均值±标准差）
- 结果输出：控制台表格、CSV、柱状图

数据说明：
- 时间：day2-day7等
- 条件：1-6 (不同培养比例条件)
- 面积分布：550-3450区间的频数分布
- 生长率范围：约-0.14到0.89
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import (
    LinearRegression,
    RidgeCV,
    LassoCV,
    ElasticNetCV,
)
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# 固定随机种子，保证可复现
SEED: int = 42
np.random.seed(SEED)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _try_read_csv_candidates(candidates: List[str]) -> pd.DataFrame:
    """尝试从多个候选路径读取CSV文件"""
    last_err = None
    for p in candidates:
        try:
            if os.path.exists(p):
                return pd.read_csv(p)
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    raise FileNotFoundError(f"未找到可用的数据文件：{candidates}")


def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载DATASET3的直方图数据和生长率数据"""
    # 根据现有脚本的路径模式
    hist_candidates = [
        r"D:\project\2-spring\DATASET3\code\predict\processed_data\histogram_all_except.csv",
        os.path.join("DATASET3", "code", "predict", "processed_data", "histogram_all_except.csv"),
        os.path.join("ds3", "processed_data", "histogram_data.csv"),
    ]
    
    mu_candidates = [
        r"D:\project\2-spring\DATASET3\code\predict\processed_data\growth_rate_all_except.csv", 
        os.path.join("DATASET3", "code", "predict", "processed_data", "growth_rate_all_except.csv"),
        os.path.join("ds3", "processed_data", "growth_rate.csv"),
    ]

    # 读取数据
    df_hist = _try_read_csv_candidates(hist_candidates)
    df_mu = _try_read_csv_candidates(mu_candidates)

    return df_hist, df_mu


def prepare_data(df_hist: pd.DataFrame, df_mu: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """构造特征矩阵，参考DATASET3的数据处理逻辑"""
    # 创建面积区间
    area_bins = np.arange(550, 3451, 100)
    
    # 统一列名（condition_value -> condition）
    df_mu_copy = df_mu.copy()
    if 'condition_value' in df_mu_copy.columns:
        df_mu_copy = df_mu_copy.rename(columns={'condition_value': 'condition'})
    
    # 创建透视表
    pivot_df = df_hist.pivot_table(
        index=["time", "condition"],
        columns="area_bin",
        values="frequency",
        aggfunc="first",
    )
    pivot_df = pivot_df.reindex(columns=area_bins)
    pivot_df = pivot_df.interpolate(axis=1)

    # 合并数据确保顺序一致
    df_merged = pd.merge(
        pivot_df.reset_index(),
        df_mu_copy[['time', 'condition', 'mu']],
        on=['time', 'condition'],
        how='inner'
    )
    
    # 重新创建pivot表（仅包含有标签的样本）
    df_hist_filtered = df_merged[['time', 'condition'] + list(area_bins)]
    pivot_df_filtered = df_hist_filtered.set_index(['time', 'condition'])
    
    # 确保数值列为float类型
    for col in area_bins:
        if col in pivot_df_filtered.columns:
            pivot_df_filtered[col] = pd.to_numeric(pivot_df_filtered[col], errors='coerce')
    
    # 获取基础特征和标签
    X = pivot_df_filtered.values.astype(np.float64)
    y = pd.to_numeric(df_merged["mu"], errors='coerce').values.astype(np.float64)

    # 时间特征映射
    time_values = df_merged['time'].values
    time_map = {}
    unique_times = sorted(df_merged['time'].unique())
    for i, time_val in enumerate(unique_times):
        # 如果是day格式，提取数字；否则直接使用索引
        if isinstance(time_val, str) and time_val.startswith('day'):
            time_map[time_val] = int(time_val.replace('day', ''))
        else:
            time_map[time_val] = i + 1
    
    time_features = np.array([time_map[t] for t in time_values], dtype=np.float64)
    time_features = time_features.reshape(-1, 1)
    
    # 条件值特征
    condition_features = pd.to_numeric(df_merged['condition'], errors='coerce').values.astype(np.float64)
    condition_features = condition_features.reshape(-1, 1)

    # 统计特征
    mean_features = np.mean(X, axis=1, keepdims=True)
    std_features = np.std(X, axis=1, keepdims=True)
    max_features = np.max(X, axis=1, keepdims=True)
    min_features = np.min(X, axis=1, keepdims=True)
    dist_range = np.max(X, axis=1, keepdims=True) - np.min(X, axis=1, keepdims=True)
    
    # 防止除零错误
    std_features_safe = np.where(std_features == 0, 1e-8, std_features)
    dist_skew = np.mean((X - mean_features) ** 3, axis=1, keepdims=True) / (std_features_safe ** 3)
    dist_kurt = (
        np.mean((X - mean_features) ** 4, axis=1, keepdims=True) / (std_features_safe ** 4) - 3
    )

    # 百分位特征
    percentiles = [25, 50, 75]
    percentile_features = np.percentile(X, percentiles, axis=1).T

    # 区间均值特征
    area_bins_split = np.array_split(X, 3, axis=1)
    bin_means = np.array([np.mean(bin, axis=1) for bin in area_bins_split]).T

    # 组合所有特征（与原脚本保持一致，不包含时间和条件特征）
    X_combined = np.hstack([
        X,  # 原始面积分布特征
        time_features,  # 时间特征
        condition_features,  # 条件值特征
        mean_features,  # 均值
        std_features,  # 标准差
        max_features,  # 最大值
        min_features,  # 最小值
        dist_range,  # 分布范围
        dist_skew,  # 偏度
        dist_kurt,  # 峰度
        percentile_features,  # 百分位数特征
        bin_means  # 面积区间特征
    ])

    return X_combined, y


def get_models() -> Dict[str, object]:
    """构建需要比较的模型集合"""
    models: Dict[str, object] = {
        "LinearRegression": LinearRegression(),
        "RidgeCV": RidgeCV(alphas=np.logspace(-3, 3, 21)),
        "LassoCV": LassoCV(alphas=None, cv=5, random_state=SEED, max_iter=10000),
        "ElasticNetCV": ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=5, random_state=SEED, max_iter=10000),
        "SVR_RBF": SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.01),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=SEED, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500, max_depth=None, random_state=SEED, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED
        ),
        "XGBoost": xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    # LightGBM（可选）
    try:
        import lightgbm as lgb  # type: ignore

        models["LightGBM"] = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
        )
    except Exception:
        print("[提示] 未安装 LightGBM，已跳过 LightGBM 对比。")

    # CatBoost（可选）
    try:
        from catboost import CatBoostRegressor  # type: ignore

        models["CatBoost"] = CatBoostRegressor(
            loss_function="RMSE",
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            random_seed=SEED,
            verbose=False,
        )
    except Exception:
        print("[提示] 未安装 CatBoost，已跳过 CatBoost 对比。")

    return models


def evaluate_models(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """固定划分 + 交叉验证，输出结果表"""
    # 固定划分（90%训练，10%测试）
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    train_ratio = 0.9
    train_size = int(len(X_shuffled) * train_ratio)
    X_train, y_train = X_shuffled[:train_size], y_shuffled[:train_size]
    X_test, y_test = X_shuffled[train_size:], y_shuffled[train_size:]

    # 评测配置
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    results: List[Dict[str, object]] = []
    models = get_models()

    for name, estimator in models.items():
        try:
            # 使用 Pipeline 将稳健标准化纳入训练流程
            pipeline = Pipeline([
                ("scaler", RobustScaler()),
                ("model", estimator),
            ])

            # 训练并测试
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            test_mae = float(mean_absolute_error(y_test, y_pred))
            test_r2 = float(r2_score(y_test, y_pred))

            # 5折 CV（在全量数据上，以获得稳健估计）
            cv_scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=kf,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            cv_rmse_mean = float((-cv_scores).mean())
            cv_rmse_std = float((-cv_scores).std())

            results.append(
                {
                    "Model": name,
                    "Test_RMSE": test_rmse,
                    "Test_MAE": test_mae,
                    "Test_R2": test_r2,
                    "CV_RMSE_Mean": cv_rmse_mean,
                    "CV_RMSE_Std": cv_rmse_std,
                }
            )

        except Exception as e:
            print(f"[警告] 模型 {name} 训练失败: {str(e)}")
            continue

    if not results:
        print("[警告] 所有模型都训练失败，返回空结果")
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results).sort_values(by="Test_RMSE").reset_index(drop=True)
    return df_results


def plot_results(df_results: pd.DataFrame, save_dir: str) -> None:
    """绘制对比图表"""
    os.makedirs(save_dir, exist_ok=True)

    # 柱状图（按 Test RMSE 排序）
    plt.figure(figsize=(12, 6))
    order = df_results.sort_values("Test_RMSE")["Model"]
    sns.barplot(
        data=df_results,
        x="Model",
        y="Test_RMSE",
        order=order,
        color="skyblue",
        edgecolor="black",
    )
    plt.xticks(rotation=35, ha="right")
    plt.title("DATASET3 模型对比 (Test RMSE)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dataset3_model_comparison_test_rmse.png"), dpi=200)
    plt.close()

    # CV RMSE 误差条
    plt.figure(figsize=(12, 6))
    plt.errorbar(
        df_results["Model"],
        df_results["CV_RMSE_Mean"],
        yerr=df_results["CV_RMSE_Std"],
        fmt="o",
        ecolor="gray",
        capsize=4,
    )
    plt.xticks(rotation=35, ha="right")
    plt.title("DATASET3 模型对比 (5-fold CV RMSE)")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dataset3_model_comparison_cv_rmse.png"), dpi=200)
    plt.close()

    # R² 对比
    plt.figure(figsize=(12, 6))
    order = df_results.sort_values("Test_R2", ascending=False)["Model"]
    sns.barplot(
        data=df_results,
        x="Model",
        y="Test_R2",
        order=order,
        color="lightgreen",
        edgecolor="black",
    )
    plt.xticks(rotation=35, ha="right")
    plt.title("DATASET3 模型对比 (Test R2)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dataset3_model_comparison_test_r2.png"), dpi=200)
    plt.close()


def main() -> None:
    print("[信息] 加载 DATASET3 数据...")
    try:
        df_hist, df_mu = load_dataset()
        print(f"[信息] 数据加载成功")
        print(f"  - 直方图数据: {len(df_hist)} 行")
        print(f"  - 生长率数据: {len(df_mu)} 行")
    except Exception as e:
        print(f"[错误] 数据加载失败: {str(e)}")
        return

    print("[信息] 构造特征...")
    try:
        X, y = prepare_data(df_hist, df_mu)
        print(f"[信息] 特征构造完成，特征维度: {X.shape}, 标签范围: [{y.min():.3f}, {y.max():.3f}]")
        
        # 检查数据类型
        print(f"[调试] X数据类型: {X.dtype}, X形状: {X.shape}")
        print(f"[调试] y数据类型: {y.dtype}, y形状: {y.shape}")
        
        # 安全检查NaN（只对数值类型）
        if np.issubdtype(X.dtype, np.number):
            print(f"[调试] X是否包含NaN: {np.isnan(X).any()}")
        else:
            print(f"[警告] X包含非数值数据类型: {X.dtype}")
            print(f"[调试] X的前几个值: {X[:3] if len(X) > 0 else 'Empty'}")
        
        if np.issubdtype(y.dtype, np.number):
            print(f"[调试] y是否包含NaN: {np.isnan(y).any()}")
        else:
            print(f"[警告] y包含非数值数据类型: {y.dtype}")
        
    except Exception as e:
        print(f"[错误] 特征构造失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    print("[信息] 开始多模型评测...")
    df_results = evaluate_models(X, y)

    if df_results.empty:
        print("[错误] 没有成功评测任何模型")
        return

    # 打印漂亮的表（仿顶会）
    pretty = df_results.copy()
    pretty["CV_RMSE_Mean+-Std"] = (
        pretty["CV_RMSE_Mean"].map(lambda v: f"{v:.4f}") + 
        " +- " + 
        pretty["CV_RMSE_Std"].map(lambda v: f"{v:.4f}")
    )
    pretty = pretty[["Model", "Test_RMSE", "Test_MAE", "Test_R2", "CV_RMSE_Mean+-Std"]]

    print("\n=== DATASET3 Benchmark Results (sorted by Test RMSE) ===")
    print(pretty.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 保存结果
    save_dir = os.path.join("DATASET3", "results")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "dataset3_model_benchmark_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[信息] 详细结果已保存：{csv_path}")

    # 绘图
    plot_results(df_results, save_dir)
    print(f"[信息] 图表已保存至：{save_dir}")

    # 输出最佳模型信息
    best_model = df_results.iloc[0]
    print(f"\n[信息] 最佳模型: {best_model['Model']}")
    print(f"  - Test RMSE: {best_model['Test_RMSE']:.4f}")
    print(f"  - Test R2: {best_model['Test_R2']:.4f}")
    print(f"  - CV RMSE: {best_model['CV_RMSE_Mean']:.4f} +- {best_model['CV_RMSE_Std']:.4f}")


if __name__ == "__main__":
    main()
