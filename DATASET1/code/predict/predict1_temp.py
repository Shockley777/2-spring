"""
藻类生长率预测模型 (温度条件版本)

输入特征说明：
1. 温度分布直方图特征 (30维)
   - 每个温度区间(550-3450)的频数分布
   - 通过pivot_table和interpolate处理得到连续分布

2. 时间特征 (1维)
   - day1到day6，转换为数字1-6
   - 表示实验进行的天数

3. 条件值特征 (1维)
   - condition_value: 22, 24, 26, 28, 30
   - 表示不同的温度条件

4. 统计特征 (6维)
   - 均值：温度分布的平均值
   - 标准差：温度分布的离散程度
   - 最大值：最高温度频数
   - 最小值：最低温度频数
   - 温度范围：最大值-最小值
   - 偏度：温度分布的对称性
   - 峰度：温度分布的尖峰程度

输出说明：
- 生长率μ：连续值，表示藻类在特定条件下的生长速率
- 范围：约0.29到1.31
- 单位：天^-1

模型评估指标：
1. RMSE：均方根误差，反映预测值与真实值的平均偏差
2. R²：决定系数，反映模型解释数据变异的程度
3. 交叉验证RMSE：模型泛化能力的评估
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')
import optuna  # 新增Optuna导入

# 设置随机种子
SEED = 28
np.random.seed(SEED)

# ------------------- 数据准备 -------------------
# 读取温度分布数据
df_hist = pd.read_csv(r"D:\project\2-spring\DATASET1\processed_data\histogram_data.csv")

# 加载生长率数据
# mu_data = {
#     "time": ["day1"] * 5 + ["day2"] * 5 + ["day3"] * 5 + ["day4"] * 5 + ["day5"] * 5 + ["day6"] * 5,
#     "condition_value": [22, 24, 26, 28, 30] * 6,
#     "mu": [0.405465108,0.810930216,1.037987667,0.737598943,1.312186389,0.725937003,0.773189888,0.672093771,0.815749503,0.730887509,
#            0.814508038,0.802346473,0.813291492,0.844546827,0.823200309,0.700264648,0.663990596,0.774640215,0.737598943,0.854242333,
#            0.349557476,0.519075523,0.52806743,0.580292691,0.610216801,0.58221562,0.490910314,0.47957308,0.440251224,0.291434422]
# }
df_mu = pd.read_csv(r"D:\project\2-spring\DATASET1\processed_data\growth_rate.csv")
df = pd.merge(df_hist, df_mu, on=["time", "condition_value"])
# print(df.head(300))
# print(df.tail(300))
# 数据预处理
def prepare_data(df):
    # 创建特征矩阵
    area_bins = np.arange(550, 3451, 100)
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
    
    # 添加时间特征
    time_map = {"day1": 1, "day2": 2, "day3": 3, "day4": 4, "day5": 5, "day6": 6}
    time_features = np.array([time_map[t] for t in pivot_df.index.get_level_values(0)])
    time_features = time_features.reshape(-1, 1)
    
    # 添加条件值特征
    condition_features = pivot_df.index.get_level_values(1).values.reshape(-1, 1)
    
    # 计算统计特征
    mean_features = np.mean(X, axis=1, keepdims=True)
    std_features = np.std(X, axis=1, keepdims=True)
    max_features = np.max(X, axis=1, keepdims=True)
    min_features = np.min(X, axis=1, keepdims=True)
    
    # 计算面积分布的特征
    dist_range = np.max(X, axis=1, keepdims=True) - np.min(X, axis=1, keepdims=True)
    dist_skew = np.mean((X - mean_features) ** 3, axis=1, keepdims=True) / (std_features ** 3)
    dist_kurt = np.mean((X - mean_features) ** 4, axis=1, keepdims=True) / (std_features ** 4) - 3
    
    # 计算面积分布的百分位数特征
    percentiles = [25, 50, 75]
    percentile_features = np.percentile(X, percentiles, axis=1).T
    
    # 计算面积分布的区间特征
    area_bins = np.array_split(X, 3, axis=1)  # 将面积范围分成3个区间
    bin_means = np.array([np.mean(bin, axis=1) for bin in area_bins]).T  # 面积区间特征
    
    # 组合所有特征
    X = np.hstack([
        X,  # 原始特征
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
    
    return X, y

X, y = prepare_data(df)

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

# ------------------- 模型训练和评估 -------------------
def train_and_evaluate():
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
    
    print(f"\n训练集 RMSE: {train_rmse:.4f}")
    print(f"测试集 RMSE: {test_rmse:.4f}")
    print(f"训练集 R²: {train_r2:.4f}")
    print(f"测试集 R²: {test_r2:.4f}")
    
    # 特征重要性分析
    importance = model.get_score(importance_type='gain')
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    # 绘制特征重要性
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
    plt.scatter(y_test, test_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('True μ Values')
    plt.ylabel('Predicted μ Values')
    plt.title(f'True vs Predicted Values (R² = {test_r2:.4f})')
    plt.grid(True)
    plt.savefig('prediction_results.png')
    plt.show()
    
    # 误差分析
    errors = np.abs(y_test - test_pred)
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=15, color='skyblue', edgecolor='black')
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Absolute Error')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig('error_distribution.png')
    plt.show()
    
    # 残差分析
    residuals = y_test - test_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(test_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (True - Predicted)')
    plt.grid(True)
    plt.savefig('residual_plot.png')
    plt.show()
    
    # 打印详细预测结果
    print("\nDetailed Predictions:")
    for i, (true, pred) in enumerate(zip(y_test, test_pred)):
        print(f"Sample {i+1}: True = {true:.4f}, Predicted = {pred:.4f}, Error = {abs(true-pred):.4f}")
    
    # 误差统计
    print("\nError Statistics:")
    print(f"Mean Absolute Error: {np.mean(errors):.4f}")
    print(f"Median Absolute Error: {np.median(errors):.4f}")
    print(f"Error Standard Deviation: {np.std(errors):.4f}")
    print(f"Maximum Error: {np.max(errors):.4f}")
    print(f"Minimum Error: {np.min(errors):.4f}")
    
    return model, test_pred

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
            'max_depth': 8,
            'learning_rate': 0.0667870147689797,
            'n_estimators': 100,
            'subsample': 0.9547159515713336,
            'colsample_bytree':0.8348069872645788,
            'min_child_weight': 4,
            'gamma': 0.012099067092256277,
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

def objective(trial):
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 2),
        'n_estimators': 100,
        'random_state': SEED
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []
    for train_idx, val_idx in kf.split(X_scaled):
        dtrain = xgb.DMatrix(X_scaled[train_idx], label=y[train_idx])
        dval = xgb.DMatrix(X_scaled[val_idx], label=y[val_idx])
        model = xgb.train(param, dtrain, num_boost_round=100,
                          evals=[(dval, 'val')],
                          early_stopping_rounds=10,
                          verbose_eval=False)
        preds = model.predict(dval)
        rmse = np.sqrt(mean_squared_error(y[val_idx], preds))
        cv_scores.append(rmse)
    return float(np.mean(cv_scores))

# ------------------- 执行流程 -------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'optuna':
        print("\n正在进行Optuna自动调参...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)  # 可调整尝试次数
        print("最优参数：", study.best_params)
        print("最优RMSE：", study.best_value)
    else:
        print("Training model...")
        model, predictions = train_and_evaluate()
        print("\nPerforming cross-validation...")
        cross_validate()