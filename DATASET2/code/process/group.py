import os
import shutil

def organize_images_in_subfolders():
    # 获取当前工作目录
    current_directory = os.getcwd()
    
    # 遍历当前目录下的所有文件和文件夹
    for day_folder in os.listdir(current_directory):
        day_folder_path = os.path.join(current_directory, day_folder)
        
        # 检查是否是文件夹，并且文件夹名称以 'DAY' 开头
        if os.path.isdir(day_folder_path) and day_folder.startswith('DAY'):
            # 遍历该文件夹内的所有文件和文件夹
            for data_folder in os.listdir(day_folder_path):
                data_folder_path = os.path.join(day_folder_path, data_folder)
                
                # 检查是否是文件夹，并且文件夹名称以 'data' 开头
                if os.path.isdir(data_folder_path) and data_folder.startswith('data'):
                    # 在该文件夹内创建 images 文件夹（如果不存在）
                    images_folder = os.path.join(data_folder_path, 'images')
                    if not os.path.exists(images_folder):
                        os.makedirs(images_folder)
                    
                    # 遍历该文件夹内的所有文件
                    for file_name in os.listdir(data_folder_path):
                        file_path = os.path.join(data_folder_path, file_name)
                        
                        # 检查是否是文件（排除子文件夹）
                        if os.path.isfile(file_path):
                            # 检查文件扩展名是否是图片格式
                            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                                # 移动文件到 images 子文件夹
                                shutil.move(file_path, os.path.join(images_folder, file_name))
                                print(f"Moved: {os.path.join(day_folder, data_folder, file_name)} -> {os.path.join(day_folder, data_folder, 'images', file_name)}")

if __name__ == "__main__":
    organize_images_in_subfolders()