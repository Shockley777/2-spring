import os
import cv2
import numpy as np

# 输入和输出文件夹
input_folder = r'D:\project\2-spring\DATASET5\data\mask'         # 这里改成你的mask图片所在文件夹
output_folder = r'D:\project\2-spring\DATASET5\data\binary_mask'

# 新建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历所有png图片
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.png'):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"无法读取 {img_path}")
            continue
        # 所有非0像素设为255
        binary_mask = np.where(img > 0, 255, 0).astype(np.uint8)
        # 保存到新文件夹
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, binary_mask)
        print(f"已处理并保存: {out_path}")

print("全部处理完成！")