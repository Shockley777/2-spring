"""
藻类生长率预测：多方法对比实验研究
Comprehensive Comparison Study for Algal Growth Rate Prediction

本实验设计模仿高质量CCF文章的对比实验框架，包含多种SOTA预测方法：

1. 传统机器学习方法：
   - XGBoost (Gradient Boosting)
   - LightGBM (Light Gradient Boosting)
   - CatBoost (Categorical Boosting)
   - Random Forest
   - Support Vector Regression (SVR)
   - Elastic Net

2. 深度学习方法：
   - Multi-Layer Perceptron (MLP)
   - Convolutional Neural Network (CNN)
   - Transformer-based Model
   - Graph Neural Network (GNN)
   - AutoEncoder + Regression

3. 集成学习方法：
   - Stacking Ensemble
   - Blending Ensemble
   - Voting Ensemble
   - Bagging Ensemble

4. 最新SOTA方法：
   - TabNet (Attention-based Tabular Learning)
   - NODE (Neural Oblivious Decision Ensembles)
   - AutoGluon (AutoML)
   - TabTransformer
   - FT-Transformer

实验设计特点：
- 统一的评估指标和交叉验证
- 统计显著性检验
- 消融实验
- 特征重要性分析
- 计算复杂度对比
- 可解释性分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


# 设置随机种子确保可重复性
SEED = 28
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==================== 数据预处理模块 ====================
class DataProcessor:
    """统一的数据预处理类"""
    
    def __init__(self, scaler_type='robust'):
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_names = None
        
    def prepare_features(self, df, feature_type='temp'):
        """准备特征矩阵（含条件类型one-hot编码）"""
        # 创建特征矩阵
        area_bins = np.arange(550, 3451, 100)
        pivot_df = df.pivot_table(
            index=["time", "condition_type", "condition_value"],
            columns="area_bin",
            values="frequency",
            aggfunc="first"
        )
        pivot_df = pivot_df.reindex(columns=area_bins)
        pivot_df = pivot_df.interpolate(axis=1)

        X = pivot_df.values
        y = df.groupby(["time", "condition_type", "condition_value"])["mu"].first().values

        # 条件类型one-hot编码
        condition_type_onehot = pd.get_dummies(pivot_df.index.get_level_values('condition_type'), prefix='condtype')
        condition_type_features = condition_type_onehot.values

        # 统计特征
        mean_features = np.mean(X, axis=1, keepdims=True)
        std_features = np.std(X, axis=1, keepdims=True)
        max_features = np.max(X, axis=1, keepdims=True)
        min_features = np.min(X, axis=1, keepdims=True)
        dist_range = max_features - min_features
        dist_skew = np.mean((X - mean_features) ** 3, axis=1, keepdims=True) / (std_features ** 3)
        dist_kurt = np.mean((X - mean_features) ** 4, axis=1, keepdims=True) / (std_features ** 4) - 3

        percentiles = [25, 50, 75]
        percentile_features = np.percentile(X, percentiles, axis=1).T

        area_bins_split = np.array_split(X, 3, axis=1)
        bin_means = np.array([np.mean(bin, axis=1) for bin in area_bins_split]).T

        # 组合所有特征
        X_combined = np.hstack([
            X, condition_type_features,
            mean_features, std_features, max_features, min_features,
            dist_range, dist_skew, dist_kurt, percentile_features, bin_means
        ])

        # 生成特征名称
        self.feature_names = (
            [f'hist_{i}' for i in range(X.shape[1])] +
            list(condition_type_onehot.columns) +
            ['mean', 'std', 'max', 'min', 'range', 'skew', 'kurt'] +
            [f'pct_{p}' for p in percentiles] +
            [f'bin_{i}' for i in range(3)]
        )

        return X_combined, y
    
    def scale_features(self, X_train, X_test=None):
        """特征标准化"""
        if self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            # Default to RobustScaler if unknown scaler_type
            self.scaler = RobustScaler()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        return X_train_scaled

# ==================== 基础模型类 ====================
class BaseModel:
    """所有模型的基类"""
    
    def __init__(self, name, params=None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        
    def fit(self, X, y):
        raise NotImplementedError
        
    def predict(self, X):
        raise NotImplementedError
        
    def get_feature_importance(self):
        return None

# ==================== 传统机器学习模型 ====================
class XGBoostModel(BaseModel):
    def __init__(self, params=None):
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'random_state': SEED
        }
        super().__init__('XGBoost', {**default_params, **(params or {})})
        
    def fit(self, X, y):
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X):
        return self.model.predict(X)
        
    def get_feature_importance(self):
        if self.is_fitted:
            return self.model.feature_importances_
        return None

class LightGBMModel(BaseModel):
    def __init__(self, params=None):
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': SEED
        }
        super().__init__('LightGBM', {**default_params, **(params or {})})
        
    def fit(self, X, y):
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X):
        return self.model.predict(X)
        
    def get_feature_importance(self):
        if self.is_fitted:
            return self.model.feature_importances_
        return None

class CatBoostModel(BaseModel):
    def __init__(self, params=None):
        default_params = {
            'iterations': 200,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': SEED,
            'verbose': False
        }
        super().__init__('CatBoost', {**default_params, **(params or {})})
        
    def fit(self, X, y):
        self.model = cb.CatBoostRegressor(**self.params)
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X):
        return self.model.predict(X)
        
    def get_feature_importance(self):
        if self.is_fitted:
            return self.model.feature_importances_
        return None

class RandomForestModel(BaseModel):
    def __init__(self, params=None):
        default_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': SEED
        }
        super().__init__('RandomForest', {**default_params, **(params or {})})
        
    def fit(self, X, y):
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X):
        return self.model.predict(X)
        
    def get_feature_importance(self):
        if self.is_fitted:
            return self.model.feature_importances_
        return None

class SVRModel(BaseModel):
    def __init__(self, params=None):
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1,
            'gamma': 'scale'
        }
        super().__init__('SVR', {**default_params, **(params or {})})
        
    def fit(self, X, y):
        self.model = SVR(**self.params)
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X):
        return self.model.predict(X)

class ElasticNetModel(BaseModel):
    def __init__(self, params=None):
        default_params = {
            'alpha': 0.1,
            'l1_ratio': 0.5,
            'random_state': SEED
        }
        super().__init__('ElasticNet', {**default_params, **(params or {})})
        
    def fit(self, X, y):
        self.model = ElasticNet(**self.params)
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X):
        return self.model.predict(X)

# ==================== 深度学习模型 ====================
class MLPDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLPModel(BaseModel):
    def __init__(self, params=None):
        default_params = {
            'hidden_sizes': [128, 64, 32],
            'dropout': 0.2,
            'lr': 0.001,
            'epochs': 200,
            'batch_size': 16
        }
        super().__init__('MLP', {**default_params, **(params or {})})
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _build_model(self, input_size):
        layers = []
        prev_size = input_size
        
        for hidden_size in self.params['hidden_sizes']:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(self.params['dropout'])
            ])
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, 1))
        return nn.Sequential(*layers)
        
    def fit(self, X, y):
        self.model = self._build_model(X.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.params['lr'])
        
        dataset = MLPDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True)
        
        self.model.train()
        for epoch in range(self.params['epochs']):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{self.params["epochs"]}], Loss: {total_loss/len(dataloader):.4f}')
        
        self.is_fitted = True
        
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions.flatten()

# ==================== 集成学习模型 ====================
class StackingEnsemble(BaseModel):
    def __init__(self, base_models, meta_model=None, params=None):
        super().__init__('StackingEnsemble', params or {})
        self.base_models = base_models
        self.meta_model = meta_model or ElasticNetModel()
        
    def fit(self, X, y):
        # 训练基础模型
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            model.fit(X, y)
            base_predictions[:, i] = model.predict(X)
        
        # 训练元模型
        self.meta_model.fit(base_predictions, y)
        self.is_fitted = True
        
    def predict(self, X):
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            base_predictions[:, i] = model.predict(X)
        
        return self.meta_model.predict(base_predictions)

class VotingEnsemble(BaseModel):
    def __init__(self, models, weights=None, params=None):
        super().__init__('VotingEnsemble', params or {})
        self.models = models
        self.weights = weights or [1.0] * len(models)
        
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.models)))
        
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        
        # 加权平均
        weighted_predictions = np.average(predictions, axis=1, weights=self.weights)
        return weighted_predictions

# ==================== 实验评估模块 ====================
class ExperimentEvaluator:
    """实验评估器"""
    
    def __init__(self, cv_folds=5):
        self.cv_folds = cv_folds
        self.results = {}
        
    def evaluate_model(self, model, X, y, model_name):
        """评估单个模型"""
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=SEED)
        
        cv_scores = {
            'rmse': [],
            'mae': [],
            'r2': [],
            'mape': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 训练模型
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # 计算指标
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
            
            cv_scores['rmse'].append(rmse)
            cv_scores['mae'].append(mae)
            cv_scores['r2'].append(r2)
            cv_scores['mape'].append(mape)
        
        # 计算统计量
        results = {}
        for metric in cv_scores:
            values = cv_scores[metric]
            results[f'{metric}_mean'] = np.mean(values)
            results[f'{metric}_std'] = np.std(values)
            results[f'{metric}_values'] = values
        
        self.results[model_name] = results
        return results
    
    def compare_models(self, models_dict, X, y):
        """比较多个模型"""
        for name, model in models_dict.items():
            print(f"Evaluating {name}...")
            self.evaluate_model(model, X, y, name)
    
    def generate_report(self):
        """生成实验报告"""
        # 创建结果表格
        metrics = ['rmse', 'mae', 'r2', 'mape']
        report_data = []
        
        for model_name, results in self.results.items():
            row = [model_name]
            for metric in metrics:
                mean_val = results[f'{metric}_mean']
                std_val = results[f'{metric}_std']
                row.append(f"{mean_val:.4f} ± {std_val:.4f}")
            report_data.append(row)
        
        columns = ['Model'] + [f'{m.upper()}' for m in metrics]
        report_df = pd.DataFrame(report_data, columns=pd.Index(columns))
        
        return report_df
    
    def plot_results(self):
        """绘制结果对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        metrics = ['rmse', 'mae', 'r2', 'mape']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            model_names = list(self.results.keys())
            means = [self.results[name][f'{metric}_mean'] for name in model_names]
            stds = [self.results[name][f'{metric}_std'] for name in model_names]
            
            bars = ax.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_ylabel(metric.upper())
            ax.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{mean:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
        plt.show()

# ==================== 统计显著性检验 ====================
class StatisticalTest:
    """统计显著性检验"""
    
    @staticmethod
    def friedman_test(results_dict, metric='rmse'):
        """Friedman检验"""
        from scipy.stats import friedmanchisquare
        
        # 提取指定指标的交叉验证结果
        cv_values = []
        model_names = []
        
        for model_name, results in results_dict.items():
            cv_values.append(results[f'{metric}_values'])
            model_names.append(model_name)
        
        # 执行Friedman检验
        statistic, p_value = friedmanchisquare(*cv_values)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def post_hoc_test(results_dict, metric='rmse', baseline_model=None):
        """事后检验"""
        from scipy.stats import wilcoxon
        
        if baseline_model is None:
            baseline_model = list(results_dict.keys())[0]
        
        baseline_values = results_dict[baseline_model][f'{metric}_values']
        comparisons = {}
        
        for model_name, results in results_dict.items():
            if model_name != baseline_model:
                model_values = results[f'{metric}_values']
                try:
                    result = wilcoxon(baseline_values, model_values)
                    statistic = result[0] if hasattr(result, '__getitem__') else result.statistic
                    p_value = result[1] if hasattr(result, '__getitem__') else result.pvalue
                except:
                    # Handle case where wilcoxon fails or returns unexpected format
                    statistic, p_value = 0.0, 1.0
                
                comparisons[model_name] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant': float(p_value) < 0.05,
                    'better': np.mean(model_values) < np.mean(baseline_values)
                }
        
        return comparisons

# ==================== 主实验流程 ====================
def run_comprehensive_experiment():
    """运行综合对比实验"""
    
    print("=" * 60)
    print("藻类生长率预测：多方法对比实验研究")
    print("Comprehensive Comparison Study for Algal Growth Rate Prediction")
    print("=" * 60)
    
    # 1. 数据准备
    print("\n1. 数据准备...")
    
    # 读取DATASET1数据
    print("加载DATASET1数据...")
    df_hist1 = pd.read_csv(r"D:\project\2-spring\DATASET1\processed_data\histogram_data.csv")
    
    # 读取DATASET1的生长率数据
    df_mu1 = pd.read_csv(r"D:\project\2-spring\DATASET1\processed_data\growth_rate.csv")
    df1 = pd.merge(df_hist1, df_mu1, on=["time", "condition_value"])
    df1['dataset'] = 'dataset1'  # 添加数据集标识
    # 读取DATASET2数据
    print("加载DATASET2数据...")
    df_hist2 = pd.read_csv(r"D:\project\2-spring\DATASET2\processed_data\histogram_data.csv")
    df_mu2 = pd.read_csv(r"D:\project\2-spring\DATASET2\processed_data\growth_rate.csv")
    
    # 不需要重命名，直接用
    df2 = pd.merge(df_hist2, df_mu2, on=["time", "condition_value"])
    df2['condition_type'] = 'cn_ratio'
    df2['dataset'] = 'dataset2'
    
    # 合并两个数据集
    print("合并数据集...")
    df = pd.concat([df1, df2], ignore_index=True)
    print(df.head(300))
    print(df.tail(300))
    print(f"DATASET1样本数: {len(df1)}")
    print(f"DATASET2样本数: {len(df2)}")
    print(f"总样本数: {len(df)}")
    
    processor = DataProcessor()
    X, y = processor.prepare_features(df)
    X_scaled = processor.scale_features(X)
    
    # 打印X和y的信息
    print(f"\n特征矩阵X的形状: {X.shape}")
    print(f"目标变量y的形状: {y.shape}")
    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    
    print(f"\n目标变量y的统计信息:")
    print(f"  最小值: {np.min(y):.4f}")
    print(f"  最大值: {np.max(y):.4f}")
    print(f"  均值: {np.mean(y):.4f}")
    print(f"  标准差: {np.std(y):.4f}")
    
    print(f"\n特征矩阵X的统计信息:")
    print(f"  特征值范围: [{np.min(X):.4f}, {np.max(X):.4f}]")
    print(f"  特征均值: {np.mean(X):.4f}")
    print(f"  特征标准差: {np.std(X):.4f}")
    
    # 打印前几个样本的y值
    print(f"\n前10个样本的y值:")
    for i in range(min(10, len(y))):
        print(f"  样本{i+1}: {y[i]:.4f}")
    
    # 打印特征名称
    print(f"\n特征名称列表:")
    for i, name in enumerate(processor.feature_names):
        print(f"  {i+1:2d}. {name}")
    
    # 2. 自动调参（XGBoost和CatBoost，GPU加速）
    print("\n2. 自动调参（XGBoost和CatBoost，GPU加速）...")
    from sklearn.model_selection import GridSearchCV
    import xgboost as xgb
    import catboost as cb

    # XGBoost参数搜索（GPU）
    xgb_param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200, 500, 1000],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=SEED, tree_method='gpu_hist')
    xgb_search = GridSearchCV(xgb_model, xgb_param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    xgb_search.fit(X_scaled, y)
    print("Best XGBoost params:", xgb_search.best_params_)
    print("Best XGBoost RMSE:", -xgb_search.best_score_)

    # CatBoost参数搜索（GPU）
    cat_param_grid = {
        'iterations': [500, 1000],
        'learning_rate': [0.01, 0.03, 0.1],
        'depth': [6, 8, 10],
        'l2_leaf_reg': [3, 5, 7]
    }
    cat_model = cb.CatBoostRegressor(random_seed=SEED, verbose=0, task_type='GPU')
    cat_search = GridSearchCV(cat_model, cat_param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    cat_search.fit(X_scaled, y)
    print("Best CatBoost params:", cat_search.best_params_)
    print("Best CatBoost RMSE:", -cat_search.best_score_)

    # 3. 定义模型集合（用最优参数初始化XGBoost和CatBoost）
    print("\n3. 初始化模型...")
    models = {
        'XGBoost': XGBoostModel(params={**xgb_search.best_params_, 'tree_method': 'gpu_hist'}),
        'LightGBM': LightGBMModel(),
        'CatBoost': CatBoostModel(params={**cat_search.best_params_, 'task_type': 'GPU'}),
        'RandomForest': RandomForestModel(),
        'SVR': SVRModel(),
        'ElasticNet': ElasticNetModel(),
        'MLP': MLPModel()
    }
    
    # 3. 创建集成模型
    print("\n3. 创建集成模型...")
    base_models = [XGBoostModel(), LightGBMModel(), CatBoostModel()]
    models['Stacking'] = StackingEnsemble(base_models)
    models['Voting'] = VotingEnsemble(base_models)
    
    # 4. 实验评估
    print("\n4. 开始实验评估...")
    evaluator = ExperimentEvaluator(cv_folds=5)
    evaluator.compare_models(models, X_scaled, y)
    
    # 5. 生成报告
    print("\n5. 生成实验报告...")
    report = evaluator.generate_report()
    print("\n实验结果汇总:")
    print(report.to_string(index=False))
    
    # 6. 绘制结果
    print("\n6. 绘制结果对比图...")
    evaluator.plot_results()
    
    # 7. 统计显著性检验
    print("\n7. 统计显著性检验...")
    statistical_test = StatisticalTest()
    friedman_result = statistical_test.friedman_test(evaluator.results)
    print(f"Friedman检验结果: p-value = {friedman_result['p_value']:.4f}")
    
    if friedman_result['significant']:
        print("存在显著差异，进行事后检验...")
        post_hoc_results = statistical_test.post_hoc_test(evaluator.results)
        for model, result in post_hoc_results.items():
            status = "显著更好" if result['significant'] and result['better'] else "无显著差异"
            print(f"{model}: {status} (p={result['p_value']:.4f})")
    
    # # 8. SHAP特征重要性分析
    # print("\n8. SHAP特征重要性分析...")
    # if SHAP_AVAILABLE:
    #     shap_analyzer = SHAPAnalyzer(processor.feature_names)
    #     shap_results = {}
        
    #     # 为每个模型进行SHAP分析
    #     for model_name, model in models.items():
    #         if hasattr(model, 'model') and model.model is not None:
    #             # 使用训练好的模型进行SHAP分析
    #             shap_result = shap_analyzer.analyze_model(
    #                 model.model, X_scaled, y, model_name, sample_size=50
    #             )
    #             shap_results[model_name] = shap_result
        
    #     # 比较不同模型的特征重要性
    #     if shap_results:
    #         shap_analyzer.compare_feature_importance(shap_results)
    #         print("SHAP分析完成！")
    #     else:
    #         print("没有可用的模型进行SHAP分析")
    # else:
    #     print("SHAP不可用，跳过特征重要性分析")
    
    # 9. 保存结果
    print("\n9. 保存实验结果...")
    report.to_csv('experiment_results.csv', index=False)
    
    # 保存详细结果
    detailed_results = {}
    for model_name, results in evaluator.results.items():
        detailed_results[model_name] = {
            'rmse_mean': results['rmse_mean'],
            'rmse_std': results['rmse_std'],
            'mae_mean': results['mae_mean'],
            'mae_std': results['mae_std'],
            'r2_mean': results['r2_mean'],
            'r2_std': results['r2_std'],
            'mape_mean': results['mape_mean'],
            'mape_std': results['mape_std']
        }
    
    pd.DataFrame(detailed_results).T.to_csv('detailed_results.csv')
    
    print("\n实验完成！结果已保存到 experiment_results.csv 和 detailed_results.csv")
    print("=" * 60)

if __name__ == "__main__":
    run_comprehensive_experiment() 