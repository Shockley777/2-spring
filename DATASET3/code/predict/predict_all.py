"""
藻类生长率预测模型

输入特征说明：
1. 面积分布直方图特征 (30维)
   - 每个面积区间(550-3450)的频数分布
   - 通过pivot_table和interpolate处理得到连续分布

2. 时间特征 (1维)
   - day2到day7，转换为数字2-7
   - 表示实验进行的天数

3. 培养比例特征 (1维)
   - condition_value: 1-6
   - 表示不同的培养比例条件

4. 统计特征 (6维)
   - 均值：面积分布的平均值
   - 标准差：面积分布的离散程度
   - 最大值：最大面积频数
   - 最小值：最小面积频数
   - 分布范围：最大值-最小值
   - 偏度：面积分布的对称性
   - 峰度：面积分布的尖峰程度

输出说明：
- 生长率μ：连续值，表示藻类在特定条件下的生长速率
- 范围：约-0.14到0.89
- 单位：天^-1

模型评估指标：
1. RMSE：均方根误差，反映预测值与真实值的平均偏差
2. R²：决定系数，反映模型解释数据变异的程度
3. 交叉验证RMSE：模型泛化能力的评估
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# ------------------- 数据准备 -------------------
df = pd.read_csv(r"D:\project\2-spring\DATASET3\predict\processed_data\histogram_all_except.csv")
def prepare_data(df):
    area_bins = np.arange(550, 3451, 100)
    pivot_df = df.pivot_table(
        index=["time", "condition"],
        columns="area_bin",
        values="frequency",
        aggfunc="first"
    )
    pivot_df = pivot_df.reindex(columns=area_bins)
    pivot_df = pivot_df.interpolate(axis=1)
    X = pivot_df.values
    df_mu = pd.read_csv(r"D:\project\2-spring\DATASET3\predict\processed_data\growth_rate_all_except.csv")
    y = df_mu["mu"].values
    mean_features = np.mean(X, axis=1, keepdims=True)
    std_features = np.std(X, axis=1, keepdims=True)
    max_features = np.max(X, axis=1, keepdims=True)
    min_features = np.min(X, axis=1, keepdims=True)
    dist_range = np.max(X, axis=1, keepdims=True) - np.min(X, axis=1, keepdims=True)
    dist_skew = np.mean((X - mean_features) ** 3, axis=1, keepdims=True) / (std_features ** 3)
    dist_kurt = np.mean((X - mean_features) ** 4, axis=1, keepdims=True) / (std_features ** 4) - 3
    percentiles = [25, 50, 75]
    percentile_features = np.percentile(X, percentiles, axis=1).T
    area_bins_split = np.array_split(X, 3, axis=1)
    bin_means = np.array([np.mean(bin, axis=1) for bin in area_bins_split]).T
    X = np.hstack([
        X,
        mean_features,
        std_features,
        max_features,
        min_features,
        dist_range,
        dist_skew,
        dist_kurt,
        percentile_features,
        bin_means
    ])
    return X, y

X, y = prepare_data(df)
feature_scaler = RobustScaler()
X_scaled = feature_scaler.fit_transform(X)
indices = np.random.permutation(len(X))
X_shuffled = X_scaled[indices]
y_shuffled = y[indices]
train_ratio = 0.9
train_size = int(len(X_shuffled) * train_ratio)
X_train = X_shuffled[:train_size]
y_train = y_shuffled[:train_size]
X_test = X_shuffled[train_size:]
y_test = y_shuffled[train_size:]

# ------------------- XGBoost训练与评估 -------------------
def train_and_evaluate():
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
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    train_pred = model.predict(dtrain)
    test_pred = model.predict(dtest)
    # 保证y_test和test_pred为numpy.ndarray类型
    y_test_np = np.asarray(y_test)
    test_pred_np = np.asarray(test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_np, test_pred_np))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test_np, test_pred_np)
    print(f"\n训练集 RMSE: {train_rmse:.4f}")
    print(f"测试集 RMSE: {test_rmse:.4f}")
    print(f"训练集 R²: {train_r2:.4f}")
    print(f"测试集 R²: {test_r2:.4f}")
    # 特征重要性分析
    importance = model.get_score(importance_type='gain')
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importance)), [x[1] for x in importance])
    plt.xticks(range(len(importance)), [x[0] for x in importance], rotation=45)
    plt.title('Feature Importance Analysis')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    # 预测结果可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_np, test_pred_np, alpha=0.6)
    plt.plot([min(y_test_np), max(y_test_np)], [min(y_test_np), max(y_test_np)], 'r--')
    plt.xlabel('True μ Values')
    plt.ylabel('Predicted μ Values')
    plt.title(f'True vs Predicted Values (R² = {test_r2:.4f})')
    plt.grid(True)
    plt.savefig('prediction_results.png')
    plt.show()
    # 误差分析
    errors = np.abs(y_test_np - test_pred_np)
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=15, color='skyblue', edgecolor='black')
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Absolute Error')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig('error_distribution.png')
    plt.show()
    # 残差分析
    residuals = y_test_np - test_pred_np
    plt.figure(figsize=(10, 6))
    plt.scatter(test_pred_np, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (True - Predicted)')
    plt.grid(True)
    plt.savefig('residual_plot.png')
    plt.show()
    # 打印详细预测结果
    print("\nDetailed Predictions:")
    for i, (true, pred) in enumerate(zip(y_test_np, test_pred_np)):
        print(f"Sample {i+1}: True = {true:.4f}, Predicted = {pred:.4f}, Error = {abs(true-pred):.4f}")
    # 误差统计
    print("\nError Statistics:")
    print(f"Mean Absolute Error: {np.mean(errors):.4f}")
    print(f"Median Absolute Error: {np.median(errors):.4f}")
    print(f"Error Standard Deviation: {np.std(errors):.4f}")
    print(f"Maximum Error: {np.max(errors):.4f}")
    print(f"Minimum Error: {np.min(errors):.4f}")
    return model, test_pred_np

# ------------------- 交叉验证 -------------------
def cross_validate():
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []
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
        print(f"Fold {fold+1} RMSE: {val_rmse:.4f}")
    print(f"\nCross-validation RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# ------------------- 执行流程 -------------------
if __name__ == "__main__":
    print("Training model...")
    model, predictions = train_and_evaluate()
    print("\nPerforming cross-validation...")
    cross_validate()