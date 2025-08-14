import os
import glob

def count_images_in_dataset():
    base_path = "DATASET5/CN"
    total_images = 0
    day_counts = {}
    
    # Check for DAY2 through DAY7
    for day_num in range(2, 8):
        day_folder = f"DAY{day_num}"
        day_path = os.path.join(base_path, day_folder)
        
        if not os.path.exists(day_path):
            print(f"{day_folder} 文件夹不存在")
            continue
            
        day_total = 0
        print(f"\n检查 {day_folder}:")
        
        # Look for data folders (data1, data2, etc.)
        data_folders = glob.glob(os.path.join(day_path, "data*"))
        
        for data_folder in sorted(data_folders):
            images_path = os.path.join(data_folder, "masks")
            
            if os.path.exists(images_path):
                # Count .jpg files
                image_files = glob.glob(os.path.join(images_path, "*.png"))
                count = len(image_files)
                day_total += count
                
                data_name = os.path.basename(data_folder)
                print(f"  {data_name}: {count} 张图片")
            else:
                print(f"  {os.path.basename(data_folder)}: images 文件夹不存在")
        
        day_counts[day_folder] = day_total
        total_images += day_total
        print(f"  {day_folder} 总计: {day_total} 张图片")
    
    print(f"\n=== 总结 ===")
    for day, count in day_counts.items():
        print(f"{day}: {count} 张图片")
    print(f"所有天数总计: {total_images} 张图片")
    
    return total_images, day_counts

if __name__ == "__main__":
    count_images_in_dataset()