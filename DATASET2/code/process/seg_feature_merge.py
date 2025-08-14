import os
import cv2
import numpy as np
import pandas as pd
import glob
import math
from cellpose import models, io
from skimage import measure
from skimage.measure import regionprops
from scipy.spatial.distance import pdist

# ------------------------ 通用工具函数 ------------------------
def imread_unicode(file_path):
    """处理中文路径的图像读取"""
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"读取图像失败: {e}")
        return None

# ------------------------ 分割模块 ------------------------
def enhance_mask(mask):
    """增强mask可视化效果"""
    mask = (mask / mask.max() * 255).astype(np.uint8)
    return cv2.applyColorMap(mask, cv2.COLORMAP_JET)

def remove_low_circularity(mask, threshold=0.3):
    """根据圆度过滤区域"""
    regions = regionprops(mask)
    for region in regions:
        if region.perimeter > 0:
            circ = (4 * math.pi * region.area) / (region.perimeter ** 2)
        else:
            circ = 0
        if circ < threshold:
            mask[mask == region.label] = 0
    return mask

def remove_boundary_objects(mask):
    """移除边界接触的对象"""
    mask = mask.astype(np.int32)
    height, width = mask.shape
    for region in regionprops(mask):
        minr, minc, maxr, maxc = region.bbox
        if minr == 0 or minc == 0 or maxr == height or maxc == width:
            mask[mask == region.label] = 0
    return mask

def segment_images(input_folder, output_mask_folder, model_type='cyto3', use_gpu=True):
    """执行细胞分割"""
    os.makedirs(output_mask_folder, exist_ok=True)
    model = models.Cellpose(model_type=model_type, gpu=use_gpu)
    
    for filename in [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]:
        img_path = os.path.join(input_folder, filename)
        if (img := imread_unicode(img_path)) is None:
            continue
        
        try:
            masks, *_ = model.eval(img, diameter=48, channels=[0, 0])
        except Exception as e:
            print(f"分割失败: {e}")
            continue
        
        masks = remove_low_circularity(masks, 0.6)
        masks = remove_boundary_objects(masks)
        
        mask_path = os.path.join(output_mask_folder, f"{os.path.splitext(filename)[0]}_mask.png")
        cv2.imwrite(mask_path, (masks * 255).astype(np.uint8))
        print(f"保存分割结果: {mask_path}")

# ------------------------ 特征提取模块 ------------------------
def compute_max_diameter(coords):
    """计算最大直径"""
    return pdist(coords, 'euclidean').max() if len(coords) >= 2 else 0

def extract_features(mask, filename):
    """从mask提取特征"""
    features = []
    new_label = 1
    for region in sorted(measure.regionprops(mask.astype(int)), key=lambda r: r.label):
        if region.area < 50:
            continue
        
        perimeter = region.perimeter
        area = region.area
        circularity = (4 * np.pi * area)/(perimeter**2) if perimeter > 0 else None
        aspect_ratio = region.major_axis_length/region.minor_axis_length if region.minor_axis_length > 0 else None
        
        features.append({
            "image": filename,
            "label": new_label,
            "area": area,
            "perimeter": perimeter,
            "aspect_ratio": aspect_ratio,
            "max_diameter": compute_max_diameter(region.coords),
            "circularity": circularity
        })
        new_label += 1
    return features

def process_masks(mask_folder, output_feature_folder):
    """处理所有mask文件"""
    os.makedirs(output_feature_folder, exist_ok=True)
    
    for filename in [f for f in os.listdir(mask_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]:
        mask_path = os.path.join(mask_folder, filename)
        if (mask := imread_unicode(mask_path)) is None:
            continue
        
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        if not (features := extract_features(mask.astype(int), filename)):
            continue
        
        csv_path = os.path.join(output_feature_folder, f"{os.path.splitext(filename)[0]}.csv")
        pd.DataFrame(features).to_csv(csv_path, index=False)
        print(f"保存特征文件: {csv_path}")

# ------------------------ 合并模块 ------------------------
def merge_csv(feature_folder, output_path):
    """合并所有CSV文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(feature_folder, '*.csv'))])
    df.insert(0, 'sequence', range(1, len(df)+1))
    df.to_csv(output_path, index=False)
    print(f"合并完成: {output_path}")

# ------------------------ 主流程 ------------------------
if __name__ == '__main__':
    # 遍历所有DAY开头的文件夹
    for day_folder in sorted(glob.glob("DAY*")):
        # 遍历每个DAY文件夹中的data开头的子文件夹
        for data_folder in sorted(glob.glob(os.path.join(day_folder, "data*"))):
            # 动态构建路径
            input_images = os.path.join(data_folder, "images")
            output_masks = os.path.join(data_folder, "masks")
            output_features = os.path.join(data_folder, "features")
            merged_csv = os.path.join(data_folder, "total", "merged.csv")
            
            # 打印当前处理进度
            print(f"\n{'='*40}")
            print(f"处理目录: {data_folder}")
            
            # 执行完整流程
            try:
                # 阶段1: 图像分割
                print(f"分割图像 -> {output_masks}")
                segment_images(input_images, output_masks)
                
                # 阶段2: 特征提取
                print(f"提取特征 -> {output_features}")
                process_masks(output_masks, output_features)
                
                # 阶段3: 合并CSV
                print(f"合并结果 -> {merged_csv}")
                merge_csv(output_features, merged_csv)
                
            except Exception as e:
                print(f"处理失败 {data_folder}: {str(e)}")
                continue

    print("\n所有数据处理完成！")