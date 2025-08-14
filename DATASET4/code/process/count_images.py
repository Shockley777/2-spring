import os
import glob

def count_images_in_dataset():
    base_path = "DATASET4"
    total_images = 0
    
    print("=== 计算DATASET4中的图片数量 ===\n")
    
    # 计算20250510文件夹中的图片
    print("1. 计算20250510文件夹:")
    path_20250510 = os.path.join(base_path, "20250510")
    if os.path.exists(path_20250510):
        count_20250510 = count_images_in_folder(path_20250510)
        total_images += count_20250510
        print(f"   20250510总计: {count_20250510} 张图片")
    else:
        print("   20250510文件夹不存在")
    
    print("\n2. 计算TIMECOURSE文件夹:")
    path_timecourse = os.path.join(base_path, "TIMECOURSE")
    if os.path.exists(path_timecourse):
        count_timecourse = count_images_in_folder(path_timecourse)
        total_images += count_timecourse
        print(f"   TIMECOURSE总计: {count_timecourse} 张图片")
    else:
        print("   TIMECOURSE文件夹不存在")
    
    print(f"\n=== 总结 ===")
    print(f"DATASET4总计: {total_images} 张图片")
    
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