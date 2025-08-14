import os
import cv2
import numpy as np
import pandas as pd
from cellpose import models, io
from skimage.measure import regionprops
import math

def imread_unicode(file_path):
    """
    读取含有中文路径的图像文件
    """
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"读取图像时出错: {e}")
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

def remove_low_circularity(mask, threshold=0.3):
    """
    根据圆度 (circularity) 移除 mask 中圆度小于阈值的区域。
    圆度计算公式为： (4 * π * area) / (perimeter^2)
    """
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
    """
    移除所有与图像边缘接触的对象，包括右下角的矩形和
    不完整的细胞（部分细胞在图像边缘被截断）。
    """
    # 确保 mask 为整数型
    mask = mask.astype(np.int32)
    regions = regionprops(mask)
    height, width = mask.shape

    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        # 如果 bbox 触及图像四周，则移除（将该区域置 0）
        if minr == 0 or minc == 0 or maxr == height or maxc == width:
            mask[mask == region.label] = 0
    return mask

def segment_and_save_masks(input_folder, output_folder, model_type='cyto3', use_gpu=True):
    """
    遍历 input_folder 中所有图像，使用 cellpose 进行细胞分割，
    并将 mask 保存到 output_folder，同时可选择性地
    移除圆度低于阈值的细胞以及贴边对象。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载 cellpose 模型（支持 GPU 加速）
    model = models.Cellpose(model_type=model_type, gpu=use_gpu)

    # 允许的图片格式
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(input_folder, filename)
            print(f"Processing {img_path}...")

            # 读取图片
            img = imread_unicode(img_path)
            if img is None:
                print(f"警告: 无法读取 {img_path}，跳过。")
                continue

            try:
                # 进行细胞分割，channels=[0,0] 表示单通道图像处理
                masks, flows, styles, diams = model.eval(img, diameter=48, channels=[0, 0])
            except Exception as e:
                print(f"分割过程中出错: {e}")
                continue

            # 1) 移除圆度小于阈值的区域（可根据需求修改 threshold）
            masks = remove_low_circularity(masks, threshold=0.6)

            # 2) 移除贴边对象，包括右下角矩形或其他边界不完整的细胞
            masks = remove_boundary_objects(masks)

            # 保存原始 mask（灰度图）
            mask_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_mask.png")
            cv2.imwrite(mask_filename, (masks * 255).astype(np.uint8))
            print(f"Saved raw mask: {mask_filename}")

            # # 保存增强版 mask（伪彩色）
            # enhanced_mask = enhance_mask(masks)
            # enhanced_mask_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_mask_enhanced.png")
            # cv2.imwrite(enhanced_mask_filename, enhanced_mask)
            # print(f"Saved enhanced mask: {enhanced_mask_filename}")

def process_all_folders(root_dir):
    """
    处理所有包含images文件夹的目录
    """
    # 遍历所有子目录
    for root, dirs, files in os.walk(root_dir):
        # 检查当前目录是否包含images文件夹
        if 'images' in dirs:
            print(f"\n处理目录: {root}")
            
            # 设置输入输出路径
            input_folder = os.path.join(root, 'images')
            output_folder = os.path.join(root, 'masks')
            
            # 执行分割
            segment_and_save_masks(input_folder, output_folder, model_type='cyto3', use_gpu=True)

if __name__ == '__main__':
    # 获取脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 处理所有包含images文件夹的目录
    process_all_folders(current_dir)
    
    print("\n所有图片处理完成！") 