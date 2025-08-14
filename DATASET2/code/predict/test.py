# import pandas as pd
# from pathlib import Path
# import math
#
# root_dir = "."  # 替换为实际路径
# output = []
#
# # 遍历文件夹并计算平均值（逻辑不变）
# for parent_folder in Path(root_dir).glob("folder_01*"):
#     for data_folder in parent_folder.glob("data*"):
#         csv_path = data_folder / "total" / "merged.csv"
#         if csv_path.exists():
#             try:
#                 df = pd.read_csv(csv_path)
#                 avg_area = df["area"].mean()
#                 output.append({
#                     "父文件夹": parent_folder.name,
#                     "数据子文件夹": data_folder.name,
#                     "Area平均值": avg_area
#                 })
#             except Exception as e:
#                 print(f"处理 {csv_path} 失败: {e}")
#
# # 将结果按五个一组分列写入 Excel
# if output:
#     # 计算总组数（每列5个）
#     num_groups = math.ceil(len(output) / 5)
#
#     # 创建空 DataFrame，列名为 组1, 组2, ...
#     columns = {f"组{i + 1}": [] for i in range(num_groups)}
#     df_excel = pd.DataFrame(columns)
#
#     # 填充数据（每列5行）
#     for idx, entry in enumerate(output):
#         group_num = idx // 5  # 确定当前组号（0-based）
#         row_in_group = idx % 5  # 确定当前行在组内的位置（0-4）
#         col_name = f"组{group_num + 1}"
#
#         # 格式化内容
#         content = (
#             f"路径: {entry['父文件夹']}/{entry['数据子文件夹']}\n"
#             f"平均值: {entry['Area平均值']:.2f}"
#         )
#         df_excel.at[row_in_group, col_name] = content
#
#     # 保存为 Excel 文件
#     excel_path = Path(root_dir) / "results_summary.xlsx"
#     df_excel.to_excel(excel_path, index=False)
#     print(f"结果已保存至: {excel_path}")
# else:
#     print("未找到有效数据。")

#
# import pandas as pd
# from pathlib import Path
# import math
#
# root_dir = "."  # 替换为实际路径
# output = []
#
# # 遍历文件夹并提取平均值（逻辑不变）
# for parent_folder in Path(root_dir).glob("DAY*"):
#     for data_folder in parent_folder.glob("data[1-6]"):
#         csv_path = data_folder / "total" / "merged.csv"
#         if csv_path.exists():
#             try:
#                 df = pd.read_csv(csv_path)
#                 avg_area = round(df["area"].mean(), 2)  # 直接保留两位小数
#                 output.append(avg_area)  # 仅存储数值
#             except Exception as e:
#                 print(f"处理 {csv_path} 失败: {e}")
#
# # 将数值按每列五个分组写入Excel
# if output:
#     # 计算总列数（每列5个）
#     num_columns = math.ceil(len(output) / 6)
#
#     # 创建空 DataFrame，列名为 列1, 列2, ...
#     df_excel = pd.DataFrame(columns=[f"列{i + 1}" for i in range(num_columns)])
#
#     # 填充数据（每列最多5行）
#     for idx, value in enumerate(output):
#         col_num = idx // 6  # 列号（0-based）
#         row_num = idx % 6  # 行号（0-4）
#         col_name = f"列{col_num + 1}"
#         df_excel.at[row_num, col_name] = value
#
#     # 保存为 Excel 文件
#     excel_path = Path(root_dir) / "数值汇总.xlsx"
#     df_excel.to_excel(excel_path, index=False)
#     print(f"结果已保存至: {excel_path}")
# else:
#     print("未找到有效数据。")

import pandas as pd
from pathlib import Path

root_dir = "."  # 替换为实际路径
output = {}

# 1. 遍历所有DAY文件夹（DAY2到DAY7）
day_folders = sorted(Path(root_dir).glob("DAY[2-7]"))  # 确保顺序为DAY2, DAY3,..., DAY7
data_names = [f"data{i}" for i in range(1, 7)]        # 固定data1~data6

# 2. 初始化二维表格（行：data1~data6，列：DAY2~DAY7）
df = pd.DataFrame(index=data_names, columns=[folder.name for folder in day_folders])

# 3. 填充数据
for day_folder in day_folders:
    day_name = day_folder.name
    # 遍历当前DAY文件夹中的data1~data6
    for data_name in data_names:
        csv_path = day_folder / data_name / "total" / "merged.csv"
        if csv_path.exists():
            try:
                df_csv = pd.read_csv(csv_path)
                avg_area = round(df_csv["area"].mean(), 2)
                df.at[data_name, day_name] = avg_area
            except KeyError:
                print(f"文件 {csv_path} 缺少 'area' 列")
            except Exception as e:
                print(f"读取 {csv_path} 失败: {e}")
        else:
            print(f"警告：{csv_path} 不存在")

# 4. 保存为Excel文件
excel_path = Path(root_dir) / "DAYS与Data交叉汇总.xlsx"
df.to_excel(excel_path)
print(f"结果已保存至: {excel_path}")