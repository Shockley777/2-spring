import os
import cv2
import numpy as np
import pandas as pd
import glob
import math
from skimage import measure
from scipy.spatial.distance import pdist
from cellpose import models

##############################
# Part 1: Image Segmentation using Cellpose
##############################
def imread_unicode(file_path):
    """
    读取含有中文路径的图像文件
    """
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"读取图像 {file_path} 时出错: {e}")
        return None

def enhance_mask(mask):
    """
    增强 mask 图像：
      1. 归一化到 0-255
      2. 应用伪彩色映射，使得细胞 mask 更加清晰可见
    """
    mask = (mask / mask.max() * 255).astype(np.uint8)
    color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    return color_mask

def remove_low_circularity(mask, threshold=0.6):
    """
    根据圆度 (circularity) 移除 mask 中圆度小于阈值的区域。
    圆度计算公式为： (4 * π * area) / (perimeter^2)
    """
    regions = measure.regionprops(mask)
    for region in regions:
        if region.perimeter > 0:
            circ = (4 * math.pi * region.area) / (region.perimeter ** 2)
        else:
            circ = 0
        if circ < threshold:
            mask[mask == region.label] = 0
    return mask

def remove_boundary_objects(mask):
    """
    移除所有与图像边缘接触的对象，包括右下角的矩形和
    不完整的细胞（部分细胞在图像边缘被截断）。
    """
    mask = mask.astype(np.int32)
    regions = measure.regionprops(mask)
    height, width = mask.shape
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        if minr == 0 or minc == 0 or maxr == height or maxc == width:
            mask[mask == region.label] = 0
    return mask

def segment_and_save_masks(input_folder, masks_folder, model_type='cyto3', use_gpu=True):
    """
    遍历 input_folder 中所有图像，使用 Cellpose 进行细胞分割，
    对分割结果严格按照以下方法处理：
      1. 移除圆度低于阈值的区域；
      2. 移除与图像边缘接触的对象。
    最终将 mask 保存到 masks_folder 中，文件名格式为 <原文件名>_mask.png
    """
    if not os.path.exists(masks_folder):
        os.makedirs(masks_folder)
    
    model = models.Cellpose(model_type=model_type, gpu=use_gpu)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(input_folder, filename)
            print(f"Processing {img_path} ...")
            img = imread_unicode(img_path)
            if img is None:
                print(f"警告: 无法读取 {img_path}，跳过。")
                continue
            try:
                masks, flows, styles, diams = model.eval(img, diameter=48, channels=[0, 0])
            except Exception as e:
                print(f"分割过程中出错: {e}")
                continue
            
            # 移除圆度低于阈值的区域
            masks = remove_low_circularity(masks, threshold=0.6)
            # 移除与图像边缘接触的对象
            masks = remove_boundary_objects(masks)
            
            mask_filename = os.path.join(masks_folder, f"{os.path.splitext(filename)[0]}_mask.png")
            cv2.imwrite(mask_filename, (masks * 255).astype(np.uint8))
            print(f"Saved raw mask: {mask_filename}")

##############################
# Part 2: Feature Extraction from Masks
##############################
def compute_max_diameter(coords):
    """
    计算区域中所有像素点对之间的最大欧氏距离，即最大直径
    """
    if len(coords) < 2:
        return 0
    return pdist(coords, metric='euclidean').max()

def extract_cell_features_from_mask(mask, image_filename):
    """
    针对单张 mask，根据区域提取细胞特征
    特征包括：
      - 重新排列后的标签 label（从1开始递增）
      - 面积 area
      - 周长 perimeter
      - 长宽比 aspect_ratio (major_axis_length / minor_axis_length)
      - 最大直径 max_diameter
      - 圆度 circularity = (4 * π * area) / (perimeter^2)
    返回特征字典列表
    """
    regions = sorted(measure.regionprops(mask), key=lambda r: r.label)
    features_list = []
    new_label = 1
    for region in regions:
        if region.area < 50:
            continue
        area = region.area
        perimeter = region.perimeter
        if region.minor_axis_length > 0:
            aspect_ratio = region.major_axis_length / region.minor_axis_length
        else:
            aspect_ratio = None
        max_diameter = compute_max_diameter(region.coords)
        if perimeter > 0:
            circularity = (4 * math.pi * area) / (perimeter ** 2)
        else:
            circularity = None
        features = {
            "image": image_filename,
            "label": new_label,
            "area": area,
            "perimeter": perimeter,
            "aspect_ratio": aspect_ratio,
            "max_diameter": max_diameter,
            "circularity": circularity
        }
        new_label += 1
        features_list.append(features)
    return features_list

def process_masks(masks_folder, output_csv_folder):
    """
    遍历 masks_folder 中所有 mask 文件，
    提取细胞特征，并保存为 CSV 文件到 output_csv_folder，
    每个 CSV 文件对应一张 mask（文件名与 mask 文件名一致）
    """
    if not os.path.exists(output_csv_folder):
        os.makedirs(output_csv_folder)
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    for filename in os.listdir(masks_folder):
        if filename.lower().endswith(valid_extensions):
            mask_path = os.path.join(masks_folder, filename)
            print(f"Processing mask {mask_path} ...")
            mask = imread_unicode(mask_path)
            if mask is None:
                print(f"警告: 无法读取 {mask_path}，请检查文件路径或文件完整性。")
                continue
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = mask.astype(np.int32)
            features_list = extract_cell_features_from_mask(mask, filename)
            if not features_list:
                print(f"未在 {filename} 中检测到有效的细胞区域。")
                continue
            csv_save_path = os.path.join(output_csv_folder, os.path.splitext(filename)[0] + ".csv")
            df = pd.DataFrame(features_list)
            df.to_csv(csv_save_path, index=False)
            print(f"Saved features from {filename} to {csv_save_path}")

##############################
# Part 3: Merge CSV Files
##############################
def merge_csv_files(input_folder, output_folder, output_file_name="merged.csv"):
    """
    合并 input_folder 下所有 CSV 文件，并将合并结果保存到 output_folder 中
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    df_list = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.insert(0, 'sequence', range(1, len(merged_df) + 1))
    output_file = os.path.join(output_folder, output_file_name)
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV file has been saved to: {output_file}")

##############################
# Part 4: Process Multiple Folders
##############################
import os

# 修改后的处理函数
def process_multiple_folders_nested(folders, subfolders):
    """
    对每个指定文件夹中的子文件夹（如 "Dark", "Light"）下的 "data" 子目录执行以下步骤：
      1. 使用 Cellpose 分割图像并保存 mask 到 <folder>/<subfolder>/masks
      2. 从 mask 中提取细胞特征，并保存到 <folder>/<subfolder>/features
      3. 合并所有 CSV 文件，并保存到 <folder>/<subfolder>/total/merged.csv
    """
    for folder in folders:
        for subfolder in subfolders:
            base_path = os.path.join(folder, subfolder)
            data_folder = os.path.join(base_path, "data")
            if not os.path.exists(data_folder):
                print(f"{data_folder} 不存在，跳过 {base_path}。")
                continue

            print(f"开始处理文件夹: {base_path}")

            masks_folder = os.path.join(base_path, "masks")
            features_folder = os.path.join(base_path, "features")
            total_folder = os.path.join(base_path, "total")

            for sub in [masks_folder, features_folder, total_folder]:
                os.makedirs(sub, exist_ok=True)

            # Step 1: 分割图像
            segment_and_save_masks(data_folder, masks_folder, model_type='cyto3', use_gpu=True)
            
            # Step 2: 提取细胞特征
            process_masks(masks_folder, features_folder)
            
            # Step 3: 合并所有 CSV 文件
            merge_csv_files(features_folder, total_folder, output_file_name="merged.csv")

            print(f"文件夹 {base_path} 的处理完成。\n")

# 执行任务
if __name__ == '__main__':
    folders = ['20250510', '20250511', '20250512 6PM', '20250512 9AM', '20250513 5PM', '20250513 9AM', '20250514 9AM', '20250514 9PM', '20250515 9AM', '20250516 9AM', '20250517 9AM', '20250518 9AM', '20250519 9AM', '20250520 9AM', '20250521 9AM']
    subfolders = [""]
    process_multiple_folders_nested(folders, subfolders)
    print("seg_feature_merge_nested.py 执行完成！")

