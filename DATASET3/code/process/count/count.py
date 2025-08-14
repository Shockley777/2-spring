import os
import glob

def count_images_in_dataset():
    """统计DATASET3中所有images文件夹的图片数量"""
    # 获取当前脚本所在目录，并构建到DATASET3/data的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, '..', '..', '..', 'data')
    
    print(f"搜索基础路径: {os.path.abspath(base_path)}")
    
    if not os.path.exists(base_path):
        print(f"错误：数据目录不存在 - {base_path}")
        return 0, {}
    
    total_images = 0
    date_group_counts = {}
    detailed_counts = []
    
    # 递归查找所有images文件夹
    for root, dirs, files in os.walk(base_path):
        if os.path.basename(root) == 'images':
            # 获取相对路径信息
            rel_path = os.path.relpath(root, base_path)
            path_parts = rel_path.split(os.sep)
            
            if len(path_parts) >= 2:
                date_group = path_parts[0]  # 例如：20250321
                specific_date = path_parts[1]  # 例如：20250322
                
                # 统计图片文件（支持多种格式）
                image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
                image_count = 0
                
                for pattern in image_patterns:
                    image_files = glob.glob(os.path.join(root, pattern))
                    image_count += len(image_files)
                
                # 记录详细信息
                detailed_counts.append({
                    'date_group': date_group,
                    'specific_date': specific_date,
                    'path': rel_path,
                    'count': image_count
                })
                
                # 累计到日期组
                if date_group not in date_group_counts:
                    date_group_counts[date_group] = 0
                date_group_counts[date_group] += image_count
                total_images += image_count
                
                print(f"  {date_group}/{specific_date}: {image_count} 张图片")
    
    # 按日期组和具体日期排序
    detailed_counts.sort(key=lambda x: (x['date_group'], x['specific_date']))
    
    # 按日期组打印汇总
    print(f"\n=== 按日期组汇总 ===")
    for date_group in sorted(date_group_counts.keys()):
        count = date_group_counts[date_group]
        print(f"{date_group}: {count} 张图片")
        
        # 显示该日期组下的详细情况
        group_details = [d for d in detailed_counts if d['date_group'] == date_group]
        for detail in group_details:
            print(f"  └─ {detail['specific_date']}: {detail['count']} 张")
    
    print(f"\n=== 总结 ===")
    print(f"找到 {len(detailed_counts)} 个images文件夹")
    print(f"共 {len(date_group_counts)} 个日期组")
    print(f"总图片数: {total_images} 张")
    
    return total_images, date_group_counts

def count_by_file_type():
    """按文件类型统计图片数量"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, '..', '..', '..', 'data')
    
    file_type_counts = {}
    
    for root, dirs, files in os.walk(base_path):
        if os.path.basename(root) == 'images':
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    if ext not in file_type_counts:
                        file_type_counts[ext] = 0
                    file_type_counts[ext] += 1
    
    print(f"\n=== 按文件类型统计 ===")
    for ext, count in sorted(file_type_counts.items()):
        print(f"{ext}: {count} 张")
    
    return file_type_counts

if __name__ == "__main__":
    print("🔍 开始统计DATASET3图片数量...")
    print("=" * 50)
    
    total, group_counts = count_images_in_dataset()
    
    print("\n" + "=" * 50)
    file_counts = count_by_file_type()
    
    print("\n" + "=" * 50)
    print("✅ 统计完成！")