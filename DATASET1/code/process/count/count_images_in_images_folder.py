import os
import csv

# 需要统计的天数和data编号
days = [f"DAY{i}" for i in range(1, 7)]
data_folders = [f"data{j}" for j in range(1, 6)]

# 支持的图片格式
types = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

results = []

for day in days:
    for data in data_folders:
        images_dir = os.path.join(day, data, 'images')
        if os.path.exists(images_dir):
            files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f)) and os.path.splitext(f)[1].lower() in types]
            count = len(files)
        else:
            count = 0
        results.append({'day': day, 'data': data, 'image_count': count})

# 输出到csv
with open('image_counts.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['day', 'data', 'image_count'])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print('统计完成，结果已保存到 image_counts.csv') 