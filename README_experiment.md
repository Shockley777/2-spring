# 藻类生长率预测：多方法对比实验研究

## 项目概述

本项目实现了一个全面的藻类生长率预测模型对比实验，模仿高质量CCF文章的实验设计，包含多种SOTA预测方法进行系统性对比。

## 实验设计特点

### 🎯 研究目标
- 系统性比较传统机器学习、深度学习和最新SOTA方法在藻类生长率预测任务上的性能
- 提供统计显著性检验和可解释性分析
- 为藻类培养优化提供科学依据

### 📊 实验方法
- **统一评估框架**：所有模型使用相同的交叉验证和评估指标
- **统计显著性检验**：Friedman检验 + 事后检验
- **多维度分析**：性能、训练时间、特征重要性、可解释性
- **消融实验**：分析不同特征组合的影响

## 模型集合

### 1. 传统机器学习方法
- **XGBoost**: 梯度提升决策树
- **LightGBM**: 轻量级梯度提升机
- **CatBoost**: 类别特征梯度提升
- **Random Forest**: 随机森林
- **SVR**: 支持向量回归
- **ElasticNet**: 弹性网络回归

### 2. 深度学习方法
- **MLP**: 多层感知机
- **TabNet**: 注意力机制的表格学习
- **TabTransformer**: 基于Transformer的表格数据建模
- **FT-Transformer**: 特征标记Transformer
- **AutoEncoder**: 自编码器 + 回归

### 3. 集成学习方法
- **Stacking**: 堆叠集成
- **Voting**: 投票集成

## 数据集

### 温度条件数据集 (DATASET1)
- **特征**: 温度分布直方图 (30维) + 时间特征 + 温度条件 + 统计特征
- **条件**: 22°C, 24°C, 26°C, 28°C, 30°C
- **时间**: day1-day6
- **样本数**: 30个样本

### 培养比例数据集 (DATASET2)
- **特征**: 面积分布直方图 (30维) + 时间特征 + 培养比例 + 统计特征
- **条件**: 培养比例1-6
- **时间**: day2-day7
- **样本数**: 36个样本

## 评估指标

### 主要指标
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **R²**: 决定系数
- **MAPE**: 平均绝对百分比误差

### 统计检验
- **Friedman检验**: 检验多个模型间是否存在显著差异
- **Wilcoxon符号秩检验**: 事后检验，比较最佳模型与其他模型

### 其他分析
- **训练时间**: 模型训练效率对比
- **特征重要性**: 模型可解释性分析
- **计算复杂度**: 模型复杂度评估

## 文件结构

```
├── comprehensive_comparison_study.py  # 基础模型实现
├── advanced_models.py                 # 高级SOTA模型实现
├── experiment_runner.py               # 实验运行器
├── README_experiment.md               # 实验说明文档
├── requirements.txt                   # 依赖包列表
└── results/                          # 实验结果目录
    ├── experiment_results_temp.csv    # 温度条件实验结果
    ├── experiment_results_cn.csv      # 培养比例实验结果
    ├── detailed_results_temp.csv      # 详细结果
    ├── detailed_results_cn.csv        # 详细结果
    └── figures/                       # 图表文件
```

## 安装和运行

### 1. 环境要求
```bash
Python >= 3.8
PyTorch >= 1.9.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0
lightgbm >= 3.3.0
catboost >= 1.0.0
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行实验
```bash
# 运行完整实验（温度条件和培养比例）
python experiment_runner.py

# 或者分别运行
python -c "
from experiment_runner import ComprehensiveExperimentRunner
runner = ComprehensiveExperimentRunner()
runner.run_comprehensive_experiment('temp')  # 温度条件
runner.run_comprehensive_experiment('cn')    # 培养比例
"
```

## 实验结果

### 预期输出
1. **控制台输出**: 实时显示实验进度和结果
2. **CSV文件**: 详细的实验结果表格
3. **图表文件**: 可视化对比图表
4. **统计报告**: 显著性检验结果

### 结果解读
- **性能排名**: 按RMSE排序的模型性能
- **统计显著性**: 哪些模型性能差异显著
- **训练效率**: 模型训练时间对比
- **特征重要性**: 哪些特征对预测最重要

## 实验特色

### 🔬 科学性
- 严格的交叉验证设计
- 统计显著性检验
- 多维度性能评估

### 📈 全面性
- 涵盖传统ML到最新SOTA方法
- 多种集成学习策略
- 深度学习和传统方法对比

### 🎨 可视化
- 雷达图显示综合性能
- 特征重要性热力图
- 训练时间对比图

### 📊 可重现性
- 固定随机种子
- 详细的实验记录
- 完整的代码实现

## 扩展实验

### 消融实验
```python
# 分析不同特征组合的影响
def ablation_study():
    feature_groups = {
        'histogram_only': [0:30],
        'statistical_only': [30:37],
        'time_condition_only': [30:32],
        'all_features': [0:42]
    }
    # 对每个特征组合训练模型
```

### 超参数优化
```python
# 使用Optuna进行超参数优化
import optuna

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500)
    }
    # 训练和评估模型
    return rmse_score
```

### 模型解释性
```python
# SHAP值分析
import shap

def explain_model(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)
```

## 论文写作建议

### 实验部分结构
1. **数据集描述**: 详细说明数据来源和特征
2. **基线方法**: 列出所有对比方法
3. **实验设置**: 交叉验证、评估指标、统计检验
4. **结果分析**: 性能对比、显著性检验、消融实验
5. **讨论**: 结果解释、局限性、未来工作

### 图表建议
- **表1**: 所有方法的性能对比表格
- **图1**: 主要评估指标的柱状图
- **图2**: 综合性能雷达图
- **图3**: 特征重要性分析
- **图4**: 训练时间对比

### 统计分析
- 使用Friedman检验确定是否存在显著差异
- 使用Wilcoxon检验进行成对比较
- 报告p值和效应量
- 使用Bonferroni校正控制多重比较

## 注意事项

1. **数据路径**: 确保数据文件路径正确
2. **内存使用**: 深度学习模型可能需要较多内存
3. **GPU加速**: 如果有GPU，会自动使用加速训练
4. **结果保存**: 实验结果会自动保存到results目录

## 联系方式

如有问题或建议，请通过以下方式联系：
- 邮箱: [your-email@example.com]
- GitHub: [your-github-username]

## 引用

如果使用本实验框架，请引用相关论文：

```bibtex
@article{your-paper-2024,
  title={Comprehensive Comparison of Machine Learning Methods for Algal Growth Rate Prediction},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
``` 