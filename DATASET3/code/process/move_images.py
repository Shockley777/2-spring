import os
import shutil
from pathlib import Path

def move_images_to_folder(root_dir):
    """
    遍历目录，在每个包含图片的子目录中创建images文件夹并移动图片
    """
    # 支持的图片格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    
    # 遍历所有子目录
    for root, dirs, files in os.walk(root_dir):
        # 跳过images文件夹
        if 'images' in root:
            continue
            
        # 检查当前目录是否包含图片
        has_images = any(file.lower().endswith(image_extensions) for file in files)
        
        if has_images:
            # 创建images文件夹
            images_folder = os.path.join(root, 'images')
            if not os.path.exists(images_folder):
                os.makedirs(images_folder)
                print(f"\n在目录 {root} 中创建images文件夹")
            
            # 移动图片
            for file in files:
                if file.lower().endswith(image_extensions):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(images_folder, file)
                    
                    # 如果目标文件已存在，添加数字后缀
                    if os.path.exists(dst_path):
                        base, ext = os.path.splitext(file)
                        counter = 1
                        while os.path.exists(dst_path):
                            new_name = f"{base}_{counter}{ext}"
                            dst_path = os.path.join(images_folder, new_name)
                            counter += 1
                    
                    try:
                        shutil.move(src_path, dst_path)
                        print(f"移动文件: {file} -> {os.path.basename(dst_path)}")
                    except Exception as e:
                        print(f"移动文件 {file} 时出错: {e}")

if __name__ == '__main__':
    # 获取脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 执行移动操作
    move_images_to_folder(current_dir)
    
    print("\n所有图片移动完成！") 