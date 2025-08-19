import os
import shutil

# 指定需要处理的时间文件夹和子文件夹类型
# folders = ["1h", "2h", "3h", "6h", "12h", "24h", "48h", "72h"]
folders = ['20250510', '20250511', '20250512 6PM', '20250512 9AM', '20250513 5PM', '20250513 9AM', '20250514 9AM', '20250514 9PM', '20250515 9AM', '20250516 9AM', '20250517 9AM', '20250518 9AM', '20250519 9AM', '20250520 9AM', '20250521 9AM']
# folders = ['0h']
subfolders = ['']
# subfolders = ["Dark", "Light"]

# 定义支持的图像扩展名
valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

for time_folder in folders:
    for sub in subfolders:
        src_folder = os.path.join(time_folder, sub)
        
        # 如果子目录不存在，跳过
        if not os.path.exists(src_folder):
            print(f"子文件夹 {src_folder} 不存在，跳过。")
            continue
        
        dest_folder = os.path.join(src_folder, "data")
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        
        for filename in os.listdir(src_folder):
            # 忽略 "data" 子文件夹中的内容
            if filename.lower() == "data":
                continue

            # 判断是否为图像文件
            if filename.lower().endswith(valid_extensions):
                src_file = os.path.join(src_folder, filename)
                
                # 如果文件名以 "图像_" 开头，则去除前缀
                if filename.startswith("图像_"):
                    new_filename = filename[3:]
                else:
                    new_filename = filename
                
                dest_file = os.path.join(dest_folder, new_filename)
                try:
                    shutil.copy2(src_file, dest_file)
                    os.remove(src_file)
                    print(f"已复制并删除 {src_file} 至 {dest_file}")
                except Exception as e:
                    print(f"复制 {src_file} 时出错: {e}")

print("所有图像复制、重命名并删除原始文件完成！")
