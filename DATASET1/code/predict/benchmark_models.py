"""
多模型基准评测（仿顶会对比表）

功能：
- 复用 DATASET1 的特征构造逻辑，基于温度直方图 + 时间/条件 + 统计特征进行建模
- 统一随机划分训练/测试（与现有脚本一致：总样本约30，训练20，测试10）
- 比较多种模型：线性/正则化/核方法/树模型/XGBoost（可选 LightGBM、CatBoost）
- 指标：Test-RMSE、Test-MAE、Test-R²、5折CV-RMSE（均值±标准差）
- 结果输出：控制台表格、CSV、柱状图

注意：
- 若未安装 LightGBM/CatBoost，将自动跳过并提示。
- 尽量保持与 `predict1_temp.py` 的数据口径一致。
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
SEED: int = 28
np.random.seed(SEED)


def _try_read_csv_candidates(candidates: List[str]) -> pd.DataFrame:
    last_err = None
    for p in candidates:
        try:
            if os.path.exists(p):
                return pd.read_csv(p)
        except Exception as e:  # 记录并继续尝试
            last_err = e
    if last_err:
        raise last_err
    raise FileNotFoundError(f"未找到可用的数据文件：{candidates}")


def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """尽量兼容多种路径位置，优先使用 predict1_temp.py 中使用的绝对路径。"""
    # 绝对路径（与 predict1_temp.py 一致）
    abs_hist = r"D:\project\2-spring\DATASET1\processed_data\histogram_data.csv"
    abs_mu = r"D:\project\2-spring\DATASET1\processed_data\growth_rate.csv"

    # 可能的相对路径兜底
    rel_hist_candidates = [
        os.path.join("DATASET1", "processed_data", "histogram_data.csv"),
        os.path.join("DATASET1", "data", "processed_data", "histogram_data.csv"),
    ]
    rel_mu_candidates = [
        os.path.join("DATASET1", "processed_data", "growth_rate.csv"),
        os.path.join("DATASET1", "data", "processed_data", "growth_rate.csv"),
    ]

    # 读取温度直方图
    df_hist = _try_read_csv_candidates([abs_hist] + rel_hist_candidates)
    # 读取生长率
    df_mu = _try_read_csv_candidates([abs_mu] + rel_mu_candidates)

    return df_hist, df_mu


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """与 `predict1_temp.py` 对齐的特征构造。"""
    area_bins = np.arange(550, 3451, 100)
    pivot_df = df.pivot_table(
        index=["time", "condition_value"],
        columns="area_bin",
        values="frequency",
        aggfunc="first",
    )
    pivot_df = pivot_df.reindex(columns=area_bins)
    pivot_df = pivot_df.interpolate(axis=1)

    X = pivot_df.values
    y = df.groupby(["time", "condition_value"])["mu"].first().values

    # 时间与条件
    time_map = {"day1": 1, "day2": 2, "day3": 3, "day4": 4, "day5": 5, "day6": 6}
    time_features = np.array([time_map[t] for t in pivot_df.index.get_level_values(0)])
    time_features = time_features.reshape(-1, 1)
    condition_features = pivot_df.index.get_level_values(1).values.reshape(-1, 1)

    # 统计特征
    mean_features = np.mean(X, axis=1, keepdims=True)
    std_features = np.std(X, axis=1, keepdims=True)
    max_features = np.max(X, axis=1, keepdims=True)
    min_features = np.min(X, axis=1, keepdims=True)
    dist_range = np.max(X, axis=1, keepdims=True) - np.min(X, axis=1, keepdims=True)
    dist_skew = np.mean((X - mean_features) ** 3, axis=1, keepdims=True) / (std_features ** 3)
    dist_kurt = (
        np.mean((X - mean_features) ** 4, axis=1, keepdims=True) / (std_features ** 4) - 3
    )

    # 百分位特征
    percentiles = [25, 50, 75]
    percentile_features = np.percentile(X, percentiles, axis=1).T

    # 区间均值特征（分三段）
    area_bins_split = np.array_split(X, 3, axis=1)
    bin_means = np.array([np.mean(bin, axis=1) for bin in area_bins_split]).T

    X = np.hstack(
        [
            X,
            time_features,
            condition_features,
            mean_features,
            std_features,
            max_features,
            min_features,
            dist_range,
            dist_skew,
            dist_kurt,
            percentile_features,
            bin_means,
        ]
    )

    return X, y


def get_models() -> Dict[str, object]:
    """构建需要比较的模型集合。"""
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
    """固定划分 + 交叉验证，输出结果表。"""
    # 固定划分（与现有脚本一致）
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    train_size = 20
    X_train, y_train = X_shuffled[:train_size], y_shuffled[:train_size]
    X_test, y_test = X_shuffled[train_size:], y_shuffled[train_size:]

    # 评测配置
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    results: List[Dict[str, object]] = []
    models = get_models()

    for name, estimator in models.items():
        # 使用 Pipeline 将稳健标准化纳入训练流程（避免 CV 泄漏）
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

    df_results = pd.DataFrame(results).sort_values(by="Test_RMSE").reset_index(drop=True)
    return df_results


def plot_results(df_results: pd.DataFrame, save_dir: str) -> None:
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
    plt.title("Model Comparison (Test RMSE)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_comparison_test_rmse.png"), dpi=200)
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
    plt.title("Model Comparison (5-fold CV RMSE)")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_comparison_cv_rmse.png"), dpi=200)
    plt.close()


def main() -> None:
    print("[信息] 加载数据...")
    df_hist, df_mu = load_dataset()
    df = pd.merge(df_hist, df_mu, on=["time", "condition_value"])

    print("[信息] 构造特征...")
    X, y = prepare_data(df)

    print("[信息] 开始多模型评测...")
    df_results = evaluate_models(X, y)

    # 打印漂亮的表（仿顶会）
    pretty = df_results.copy()
    pretty["CV_RMSE_Mean±Std"] = pretty["CV_RMSE_Mean"].map(lambda v: f"{v:.4f}") + " ± " + pretty["CV_RMSE_Std"].map(lambda v: f"{v:.4f}")
    pretty = pretty[["Model", "Test_RMSE", "Test_MAE", "Test_R2", "CV_RMSE_Mean±Std"]]

    print("\n=== Benchmark Results (sorted by Test RMSE) ===")
    print(pretty.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 保存结果
    save_dir = os.path.join("DATASET1", "results")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "model_benchmark_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[信息] 详细结果已保存：{csv_path}")

    # 绘图
    plot_results(df_results, save_dir)
    print(f"[信息] 图表已保存至：{save_dir}")


if __name__ == "__main__":
    main()


