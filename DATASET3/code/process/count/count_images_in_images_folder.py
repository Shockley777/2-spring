import os
import csv
import glob

# DATASET3使用日期格式的目录结构
# 结构：DATASET3/data/年月日/具体日期/images/

# 支持的图片格式
types = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

results = []

# 获取当前脚本所在目录，并构建到DATASET3/data的路径
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', '..', '..', 'data')

print(f"搜索目录: {os.path.abspath(data_dir)}")

# 递归查找所有images文件夹
for root, dirs, files in os.walk(data_dir):
    if os.path.basename(root) == 'images':
        # 获取相对路径信息
        rel_path = os.path.relpath(root, data_dir)
        path_parts = rel_path.split(os.sep)
        
        if len(path_parts) >= 2:
            date_group = path_parts[0]  # 例如：20250321
            specific_date = path_parts[1]  # 例如：20250322
            
            # 统计该images文件夹中的图片数量
            image_files = [f for f in files if os.path.splitext(f)[1].lower() in types]
            count = len(image_files)
            
            results.append({
                'date_group': date_group,
                'specific_date': specific_date,
                'full_path': rel_path,
                'image_count': count
            })
            
            print(f"{date_group}/{specific_date}: {count} 张图片")

# 按日期组和具体日期排序
results.sort(key=lambda x: (x['date_group'], x['specific_date']))

# 输出到csv
output_file = os.path.join(script_dir, 'image_counts.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['date_group', 'specific_date', 'full_path', 'image_count'])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

# 统计汇总
total_images = sum(row['image_count'] for row in results)
date_group_totals = {}
for row in results:
    group = row['date_group']
    if group not in date_group_totals:
        date_group_totals[group] = 0
    date_group_totals[group] += row['image_count']

print(f'\n=== 汇总统计 ===')
for group, total in sorted(date_group_totals.items()):
    print(f"{group}: {total} 张图片")
print(f"总计: {total_images} 张图片")

print(f'\n统计完成，结果已保存到 {output_file}') 