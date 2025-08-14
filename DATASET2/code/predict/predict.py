# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
# import xgboost as xgb
# import matplotlib.pyplot as plt
#
#
# # --------------------------
# # 处理CN条件数据 (data_CN.xlsx)
# # --------------------------
# def process_cn_data():
#     # 读取数据
#     df = pd.read_excel("data_CN.xlsx", sheet_name="combine")
#
#     # 清洗数据：删除重复的day1行和空行
#     df = df[df.iloc[:, 0].str.startswith('day', na=False)]  # 保留以day开头的行
#     df = df.drop_duplicates(subset=[df.columns[0]], keep='first')  # 修复此处语法
#
#     # 提取特征和标签
#     X = df.iloc[:, 1:6].values  # B-F列 (Mean, Std, Count, Median, peak value)
#     y = df.iloc[:, 9].values  # J列 (生长率)
#     conditions = df.iloc[:, 12].values  # M列 (实验条件)
#
#     # 独热编码实验条件
#     encoder = OneHotEncoder(sparse_output=False)
#     conditions_encoded = encoder.fit_transform(conditions.reshape(-1, 1))
#
#     # 合并特征
#     X_combined = np.hstack([X, conditions_encoded])
#
#     # 标准化
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_combined)
#
#     return X_scaled, y, encoder, scaler
#
#
# # --------------------------
# # 后续代码与之前相同...
# # --------------------------
#
#
# # 训练CN条件模型
# X_cn, y_cn, encoder_cn, scaler_cn = process_cn_data()
# X_train_cn, X_test_cn, y_train_cn, y_test_cn = train_test_split(X_cn, y_cn, test_size=0.2, random_state=42)
#
# model_cn = xgb.XGBRegressor(
#     objective='reg:squarederror',
#     n_estimators=150,
#     max_depth=4,
#     learning_rate=0.1
# )
# model_cn.fit(X_train_cn, y_train_cn)
#
#
# # --------------------------
# # 处理温度条件数据 (data_temp.xlsx)
# # --------------------------
# def process_temp_data():
#     # 读取数据
#     df = pd.read_excel("data_temp.xlsx", sheet_name="combine")
#
#     # 填充温度标签
#     current_temp = None
#     temp_labels = []
#     for idx, row in df.iterrows():
#         if str(row.iloc[12]) != 'nan':  # M列
#             current_temp = row.iloc[12]
#         temp_labels.append(current_temp)
#     df['temp'] = temp_labels
#     df = df[df['temp'].notna()]
#
#     # 提取特征和标签
#     X = df.iloc[:, 1:6].values  # B-F列
#     y = df.iloc[:, 9].values  # J列
#     conditions = df['temp'].values
#
#     # 独热编码温度条件
#     encoder = OneHotEncoder(sparse_output=False)
#     conditions_encoded = encoder.fit_transform(conditions.reshape(-1, 1))
#
#     # 合并特征
#     X_combined = np.hstack([X, conditions_encoded])
#
#     # 标准化
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_combined)
#
#     return X_scaled, y, encoder, scaler
#
#
# # 训练温度条件模型
# X_temp, y_temp, encoder_temp, scaler_temp = process_temp_data()
# X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
#     X_temp, y_temp, test_size=0.2, random_state=42
# )
#
# model_temp = xgb.XGBRegressor(
#     objective='reg:squarederror',
#     n_estimators=150,
#     max_depth=4,
#     learning_rate=0.1
# )
# model_temp.fit(X_train_temp, y_train_temp)
#
#
# # --------------------------
# # 模型评估与可视化
# # --------------------------
# def evaluate_model(model, X_test, y_test, name):
#     y_pred = model.predict(X_test)
#     r2 = r2_score(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     print(f"\n{name}模型评估:")
#     print(f"R² Score: {r2:.3f}")
#     print(f"MSE: {mse:.3f}")
#
#     # 特征重要性可视化
#     plt.figure(figsize=(10, 4))
#     feat_importances = model.feature_importances_
#     features = ['Mean', 'Std', 'Count', 'Median', 'peak'] + \
#                list(encoder.get_feature_names_out())
#     plt.barh(features, feat_importances)
#     plt.title(f"{name}特征重要性")
#     plt.show()
#
#
# # 评估CN模型
# evaluate_model(model_cn, X_test_cn, y_test_cn, "氮源/C/N条件")
#
# # 评估温度模型
# evaluate_model(model_temp, X_test_temp, y_test_temp, "温度条件")
#
#
# # --------------------------
# # 预测示例
# # --------------------------
# def predict_cn(mean, std, count, median, peak, condition):
#     # 编码实验条件
#     condition_encoded = encoder_cn.transform([[condition]])
#     # 构建特征
#     raw_features = np.array([[mean, std, count, median, peak]])
#     combined = np.hstack([raw_features, condition_encoded])
#     # 标准化
#     scaled = scaler_cn.transform(combined)
#     return model_cn.predict(scaled)[0]
#
#
# # 示例：预测C/N 24:1在指定参数下的生长率
# sample_pred = predict_cn(
#     mean=1855.91,
#     std=514.64,
#     count=7016,
#     median=1811,
#     peak=1791,
#     condition="乙酸铵(C/N 24:1)"
# )
# print(f"\n预测生长率示例（C/N 24:1）: {sample_pred:.3f}")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt


# --------------------------
# 处理CN条件数据
# --------------------------
def process_cn_data():
    df = pd.read_excel("data_CN.xlsx", sheet_name="combine")
    df = df[df.iloc[:, 0].str.startswith('day', na=False)]
    df = df.dropna(subset=[df.columns[9]])  # 删除生长率缺失的行
    df = df.drop_duplicates(subset=[df.columns[0]], keep='first')

    X = df.iloc[:, 1:6].values
    y = df.iloc[:, 9].values.astype(float)
    conditions = df.iloc[:, 12].values

    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("CN条件数据中的生长率包含无效值！")

    encoder = OneHotEncoder(sparse_output=False)
    conditions_encoded = encoder.fit_transform(conditions.reshape(-1, 1))
    X_combined = np.hstack([X, conditions_encoded])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    return X_scaled, y, encoder, scaler


# --------------------------
# 处理温度条件数据
# --------------------------
def process_temp_data():
    df = pd.read_excel("data_temp.xlsx", sheet_name="combine")

    current_temp = None
    temp_labels = []
    for idx, row in df.iterrows():
        cell_value = str(row.iloc[12]).strip()
        if cell_value not in ['nan', 'None', '']:
            current_temp = cell_value
        temp_labels.append(current_temp)
    df['temp'] = temp_labels
    df = df[df['temp'].notna()]
    df = df.dropna(subset=[df.columns[9]])  # 删除生长率缺失的行

    X = df.iloc[:, 1:6].values
    y = df.iloc[:, 9].values.astype(float)
    conditions = df['temp'].values

    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("温度条件数据中的生长率包含无效值！")

    encoder = OneHotEncoder(sparse_output=False)
    conditions_encoded = encoder.fit_transform(conditions.reshape(-1, 1))
    X_combined = np.hstack([X, conditions_encoded])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    return X_scaled, y, encoder, scaler


# --------------------------
# 训练模型
# --------------------------
try:
    # 处理CN数据
    X_cn, y_cn, encoder_cn, scaler_cn = process_cn_data()
    X_train_cn, X_test_cn, y_train_cn, y_test_cn = train_test_split(X_cn, y_cn, test_size=0.2, random_state=42)
    model_cn = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, max_depth=4, learning_rate=0.1)
    model_cn.fit(X_train_cn, y_train_cn)

    # 处理温度数据
    X_temp, y_temp, encoder_temp, scaler_temp = process_temp_data()
    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, y_temp, test_size=0.2,
                                                                            random_state=42)
    model_temp = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, max_depth=4, learning_rate=0.1)
    model_temp.fit(X_train_temp, y_train_temp)

except ValueError as e:
    print(f"数据错误: {e}")
    exit()

# --------------------------
# 评估与输出
# --------------------------
print("\nCN条件模型 R²:", model_cn.score(X_test_cn, y_test_cn))
print("温度条件模型 R²:", model_temp.score(X_test_temp, y_test_temp))