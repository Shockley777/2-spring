# 微藻数据集相似度分析工具

本项目为三个微藻数据集（DATASET1、DATASET2、DATASET3）提供了完整的相似度分析工具，用于分析不同时间点微藻细胞面积分布的相似性。

## 📁 数据集结构

### DATASET1
- **时间范围**: DAY1 - DAY6
- **数据子文件夹**: data1, data2, data3, data4, data5
- **文件路径**: `DAY{1-6}/data{1-5}/total/merged.csv`

### DATASET2  
- **时间范围**: DAY0 - DAY7
- **数据子文件夹**: data1, data2, data3, data4, data5, data6, data7, data8
- **文件路径**: `DAY{0-7}/data{1-8}/total/merged.csv`

### DATASET3
- **时间范围**: 2025年3月21日 - 2025年4月28日
- **主文件夹**: 20250321, 20250410, 20250414, 20250421
- **子文件夹**: 每个主文件夹下有多个日期子文件夹
- **文件路径**: `{主文件夹}/{子文件夹}/total/merged.csv`

## 🚀 快速开始

### 1. 环境准备

确保已安装以下Python包：
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn openpyxl
```

### 2. 数据预处理

在运行相似度分析之前，请确保已经完成了以下步骤：

1. **图像分割**: 使用Cellpose进行细胞分割
2. **特征提取**: 提取细胞形态学特征（面积、周长等）
3. **数据合并**: 将特征数据合并为CSV文件

如果还没有完成这些步骤，请先运行相应的处理脚本。

### 3. 运行相似度分析

#### 方法一：使用简化运行脚本（推荐）

```bash
# DATASET1
cd DATASET1
python run_similarity_analysis.py

# DATASET2  
cd DATASET2
python run_similarity_analysis.py

# DATASET3
cd DATASET3
python run_similarity_analysis.py
```

#### 方法二：直接运行分析脚本

```bash
# DATASET1
cd DATASET1
python similarity_analysis.py

# DATASET2
cd DATASET2  
python similarity_analysis.py

# DATASET3
cd DATASET3
python similarity_analysis.py
```

## 📊 分析结果

每个数据集的分析都会生成以下结果文件（保存在 `results/` 文件夹中）：

### 1. 数据统计
- **`cell_count_summary.xlsx`**: 各样本的细胞数量统计
  - 总细胞数
  - 在分析范围内的细胞数（500-3500像素）

### 2. 相似度分析
- **`histogram_similarity_results.xlsx`**: 详细的相似度分析结果
  - 余弦相似度 (Cosine Similarity)
  - 皮尔逊相关系数 (Pearson Correlation)  
  - 直方图交集 (Histogram Intersection)
  - 卡方距离 (Chi-Square Distance)
  - KL散度 (KL Divergence)
  - Wasserstein距离 (Wasserstein Distance)

### 3. 可视化图表
- **`all_histograms_smoothed.png`**: 所有样本的平滑直方图对比
- **`clustering_similarity_metrics.png`**: 相似度指标的层次聚类热力图
- **`clustering_distance_metrics.png`**: 距离指标的层次聚类热力图

## ⚙️ 参数配置

可以在各数据集的 `similarity_analysis.py` 文件中修改以下参数：

```python
# 参考样本设置
REFERENCE_FOLDER = 'DAY3'  # 参考文件夹
REFERENCE_DATA = 'data1'   # 参考数据子文件夹

# 分析参数
AREA_COL = "area"          # 面积列名
RATIO_OTHER = 0.5          # 其他样本采样比例
RATIO_REF = 0.6            # 参考样本采样比例
BINS = 60                  # 直方图分箱数
RANGE = (500, 3500)        # 分析的面积范围
TOP_K = 5                  # 显示最相似的前K个样本
```

## 🔍 相似度指标说明

### 相似度指标（值越高越相似）
1. **余弦相似度**: 衡量两个向量的夹角余弦值
2. **皮尔逊相关系数**: 衡量两个分布的线性相关性
3. **直方图交集**: 衡量两个直方图的重叠程度

### 距离指标（值越低越相似）
1. **卡方距离**: 基于卡方统计量的距离度量
2. **KL散度**: 衡量两个概率分布的差异
3. **Wasserstein距离**: 基于最优传输理论的距离度量

## 📈 结果解读

### 1. 直方图对比图
- 黑色粗线：参考样本的分布曲线
- 彩色线条：其他样本的分布曲线
- 曲线越接近，相似度越高

### 2. 聚类热力图
- **相似度热力图**: 红色越深表示相似度越高
- **距离热力图**: 蓝色越深表示距离越小（越相似）
- 聚类结果显示了样本间的相似性分组

### 3. Excel结果表
- 按直方图交集排序，最相似的样本排在前面
- 可以查看各种相似度指标的具体数值

## 🛠️ 故障排除

### 常见问题

1. **数据文件不存在**
   ```
   ❌ 数据文件不存在: DAY1/data1/total/merged.csv
   ```
   **解决方案**: 确保已经运行过数据预处理脚本

2. **依赖包缺失**
   ```
   ❌ 缺少依赖包: No module named 'pandas'
   ```
   **解决方案**: 安装缺失的包
   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn openpyxl
   ```

3. **内存不足**
   **解决方案**: 减少采样比例或分箱数
   ```python
   RATIO_OTHER = 0.3  # 从0.5减少到0.3
   BINS = 40          # 从60减少到40
   ```

## 📝 注意事项

1. **数据质量**: 确保CSV文件包含有效的面积数据
2. **内存使用**: 大数据集可能需要较多内存
3. **计算时间**: 样本数量越多，计算时间越长
4. **结果解释**: 相似度分析结果需要结合实验背景进行解释

## 🤝 贡献

如有问题或建议，请提交Issue或Pull Request。

## 📄 许可证

本项目采用MIT许可证。 