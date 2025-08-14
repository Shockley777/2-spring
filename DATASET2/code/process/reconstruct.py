import os
import shutil

root_dir = "."  # 根目录路径

# 遍历 DAY2 到 DAY7
for day in range(2, 8):
    day_dir = os.path.join(root_dir, f"DAY{day}")

    # 检查 DAY 文件夹是否存在
    if not os.path.exists(day_dir):
        print(f"警告：{day_dir} 不存在，跳过")
        continue

    # 遍历 DAYX 下的所有 data* 文件夹（data1到data6）
    for data_num in range(1, 7):
        data_folder = os.path.join(day_dir, f"data{data_num}")
        src_file = os.path.join(data_folder, "total", "merged.csv")

        # 检查源文件是否存在
        if not os.path.exists(src_file):
            print(f"警告：{src_file} 不存在，跳过")
            continue

        # 目标路径：根目录下的 dataY/dayX.csv
        dest_folder = os.path.join(root_dir, f"data{data_num}")
        dest_file = os.path.join(dest_folder, f"day{day}.csv")

        # 创建目标文件夹（若不存在）
        os.makedirs(dest_folder, exist_ok=True)

        # 复制并重命名文件
        shutil.copy2(src_file, dest_file)
        print(f"成功提取 DAY{day}/data{data_num} 的 merged.csv 到 data{data_num}/day{day}.csv")