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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ------------------- 数据准备 -------------------
# 读取温度分布数据
df_hist = pd.read_csv("CN\processed_data\histogram_data.csv")

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
    time_map = {"day2": 2, "day3": 3, "day4": 4, "day5": 5, "day6": 6, "day7": 7}
    time_features = np.array([time_map[t] for t in pivot_df.index.get_level_values(0)])
    time_features = time_features.reshape(-1, 1)
    
    # 添加培养比例特征
    condition_features = pivot_df.index.get_level_values(1).values.reshape(-1, 1)
    
    # 计算面积分布的基本统计特征
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
    bin_means = np.array([np.mean(bin, axis=1) for bin in area_bins]).T  # 修改这里，确保输出是2维的
    
    # 组合所有特征
    X = np.hstack([
        X,  # 原始特征
        time_features,  # 时间特征
        condition_features,  # 培养比例特征
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

# ------------------- 数据集定义 -------------------
class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建数据加载器
train_dataset = RegressionDataset(X_train, y_train)
test_dataset = RegressionDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# ------------------- 简化的模型定义 -------------------
class SimpleRegressionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

# 创建模型
model = SimpleRegressionModel(input_size=X.shape[1])

# ------------------- 训练配置 -------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# ------------------- 训练循环 -------------------
def train_model():
    best_loss = float("inf")
    train_losses = []
    val_losses = []
    patience = 20  # 增加早停的耐心值
    patience_counter = 0

    for epoch in range(100):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证集评估
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        # 记录损失值
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(test_loader)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # 学习率调整
        scheduler.step(epoch_val_loss)

        # 保存最佳模型
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

        print(f"Epoch {epoch + 1:03d} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("Training & Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()

# ------------------- 预测和可视化 -------------------
def visualize_predictions(loader):
    model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            all_targets.extend(targets.numpy())
            all_preds.extend(outputs.numpy())

    targets = np.array(all_targets).flatten()
    preds = np.array(all_preds).flatten()

    # 计算R²分数
    r2 = r2_score(targets, preds)
    
    # 散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, preds, alpha=0.6)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel("True μ Values")
    plt.ylabel("Predicted μ Values")
    plt.title(f"True vs. Predicted Values (R² = {r2:.4f})")
    plt.grid(True)
    plt.savefig("true_vs_pred.png")
    plt.show()

    # 打印详细的预测结果
    print("\nDetailed Predictions:")
    for i, (true, pred) in enumerate(zip(targets, preds)):
        print(f"Sample {i+1}: True = {true:.4f}, Predicted = {pred:.4f}, Error = {abs(true-pred):.4f}")

    # 绘制误差分布图
    errors = np.abs(targets - preds)
    plt.figure(figsize=(10, 5))
    sns.histplot(errors, bins=20)
    plt.title("Distribution of Prediction Errors")
    plt.xlabel("Absolute Error")
    plt.ylabel("Count")
    plt.savefig("error_distribution.png")
    plt.show()

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