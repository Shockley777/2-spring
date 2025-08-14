import os
import glob

def count_images_in_dataset():
    base_path = "DATASET3"
    total_images = 0
    
    print("=== 计算DATASET3中的图片数量 ===\n")
    
    # 获取所有主要文件夹
    main_folders = ["20250321", "20250410", "20250414", "20250421"]
    
    for folder in main_folders:
        print(f"计算 {folder} 文件夹:")
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            count = count_images_in_folder(folder_path)
            total_images += count
            print(f"   {folder} 总计: {count} 张图片\n")
        else:
            print(f"   {folder} 文件夹不存在\n")
    
    print(f"=== 总结 ===")
    print(f"DATASET3总计: {total_images} 张图片")
    
    return total_images

def count_images_in_folder(folder_path):
    """递归计算文件夹中所有.jpg图片的数量"""
    total_count = 0
    
    # 遍历所有子文件夹
    for root, dirs, files in os.walk(folder_path):
        # 检查当前文件夹是否包含images子文件夹
        images_path = os.path.join(root, "images")
        if os.path.exists(images_path):
            # 计算images文件夹中的.jpg文件数量
            image_files = glob.glob(os.path.join(images_path, "*.jpg"))
            count = len(image_files)
            if count > 0:
                # 显示相对路径
                rel_path = os.path.relpath(root, folder_path)
                print(f"   {rel_path}/images: {count} 张图片")
                total_count += count
    
    return total_count

if __name__ == "__main__":
    count_images_in_dataset() 