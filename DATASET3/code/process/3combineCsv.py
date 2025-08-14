import os
import glob
import pandas as pd

def process_folder(folder_path):
    # 定义文件夹路径
    input_folder = os.path.join(folder_path, 'features')
    output_folder = os.path.join(folder_path, 'total')
    output_file = os.path.join(output_folder, 'merged.csv')

    # 检查features文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"Features folder not found in {folder_path}")
        return

    # 获取所有CSV文件的路径
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return

    # 读取所有CSV文件，并存储在列表中
    df_list = [pd.read_csv(file) for file in csv_files]

    # 合并所有DataFrame，忽略原有索引
    merged_df = pd.concat(df_list, ignore_index=True)

    # 新增一列表示数据序号，从1开始
    merged_df.insert(0, 'sequence', range(1, len(merged_df) + 1))

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 将合并后的DataFrame保存为CSV文件，不保存行索引
    merged_df.to_csv(output_file, index=False)

    print(f"合并后的文件已保存至: {output_file}")

def main():
    # 获取脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 获取5_8目录的路径
    base_dir = os.path.join(os.path.dirname(current_dir), '5_8')
    
    # 遍历5_8目录下的所有子目录
    for root, dirs, files in os.walk(base_dir):
        # 检查当前目录是否同时包含images和features文件夹
        if 'images' in dirs and 'features' in dirs:
            print(f"\n处理目录: {root}")
            process_folder(root)

if __name__ == '__main__':
    main()
    print("\n所有文件夹处理完成！") 