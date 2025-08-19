import os
import cv2
import numpy as np
import pandas as pd
from skimage import measure
from scipy.spatial.distance import pdist

def imread_unicode(file_path):
    """
    使用 np.fromfile 和 cv2.imdecode 处理中文路径问题
    """
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        # 使用 cv2.IMREAD_UNCHANGED 保留原图通道信息
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        print(f"读取图像 {file_path} 时出错: {e}")
        return None

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
      - 圆度 circularity = (4π * area) / (perimeter^2)
    返回特征字典列表
    """
    # 获取所有区域，并按原label升序排列
    regions = sorted(measure.regionprops(mask), key=lambda r: r.label)
    features_list = []
    new_label = 1  # 新的label从1开始
    for region in regions:
        # 过滤面积过小的区域（可能为噪声），将阈值设为50
        if region.area < 50:
            continue
        
        area = region.area
        perimeter = region.perimeter
        
        # 计算长宽比
        if region.minor_axis_length > 0:
            aspect_ratio = region.major_axis_length / region.minor_axis_length
        else:
            aspect_ratio = None

        max_diameter = compute_max_diameter(region.coords)

        # 计算圆度，确保周长不为0
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
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

    # 定义支持的图像扩展名
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    for filename in os.listdir(masks_folder):
        if filename.lower().endswith(valid_extensions):
            mask_path = os.path.join(masks_folder, filename)
            print(f"Processing {mask_path}...")
            
            mask = imread_unicode(mask_path)
            if mask is None:
                print(f"警告: 无法读取 mask 文件 {mask_path}，请检查文件路径或文件完整性。")
                continue

            # 若 mask 为多通道，则转换为单通道
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # 确保 mask 为整数类型
            mask = mask.astype(np.int32)
            
            features_list = extract_cell_features_from_mask(mask, filename)
            if not features_list:
                print(f"未在 {filename} 中检测到有效的细胞区域。")
                continue

            # 保存 CSV 文件
            csv_save_path = os.path.join(output_csv_folder, os.path.splitext(filename)[0] + ".csv")
            df = pd.DataFrame(features_list)
            df.to_csv(csv_save_path, index=False)
            print(f"保存 {filename} 的特征到 {csv_save_path}")

if __name__ == '__main__':
    masks_folder = "data1/masks"       # mask 文件夹路径
    output_csv_folder = "data1/features"  # CSV 文件输出路径
    process_masks(masks_folder, output_csv_folder)
