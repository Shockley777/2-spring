#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import cv2
import numpy as np
import pandas as pd
from skimage import measure
from scipy.spatial.distance import pdist
from multiprocessing import Pool, cpu_count
from functools import partial

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
    计算区域的最大直径（Feret max 近似）。
    性能优化：先对像素点做凸包，再在凸包顶点上计算两两距离的最大值，
    将复杂度从 O(N^2) 降到 O(M^2)（M 为凸包顶点数，远小于像素数）。
    """
    if coords is None or len(coords) < 2:
        return 0.0
    try:
        # region.coords 是 (row, col)，转为 (x, y)
        pts = np.asarray(coords, dtype=np.int32)[:, ::-1].astype(np.float32)
        if pts.shape[0] < 2:
            return 0.0
        hull = cv2.convexHull(pts)
        hull_pts = hull.reshape(-1, 2)
        if hull_pts.shape[0] < 2:
            return 0.0
        # 在凸包顶点上计算最大两点距离
        dists = pdist(hull_pts, metric='euclidean')
        return float(dists.max()) if dists.size > 0 else 0.0
    except Exception:
        # 退化回原始实现（极少触发）
        try:
            return float(pdist(coords, metric='euclidean').max())
        except Exception:
            return 0.0

def extract_cell_features_from_mask(mask, image_filename, nuclei_mask=None, rgb_img=None):
    """
    针对单张 mask，根据区域提取细胞特征（面向梭形/细长细胞，移除圆度）。
    输出特征：
      - label（从1开始重排）
      - area, perimeter
      - major_axis_length, minor_axis_length, aspect_ratio
      - eccentricity, orientation_rad, orientation_deg
      - centroid_y, centroid_x
      - max_diameter（像素坐标对最大欧氏距离，Feret max 近似）
      - bbox_area, extent (= area / bbox_area)
      - solidity (= area / convex_area)
      - equivalent_diameter（基于面积等效直径，若可用）
      "nuclear_area": nuclear_area,
            "nuclear_fraction": nuclear_fraction,
            "nuclear_mean_R": nuclear_mean_R,
            "nuclear_mean_G": nuclear_mean_G,
            "nuclear_mean_B": nuclear_mean_B,
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
        
        # 轴长与长宽比
        major_axis_length = getattr(region, 'major_axis_length', None)
        minor_axis_length = getattr(region, 'minor_axis_length', None)
        if minor_axis_length and minor_axis_length > 0:
            aspect_ratio = major_axis_length / minor_axis_length
        else:
            aspect_ratio = None

        # 形状与方向
        eccentricity = getattr(region, 'eccentricity', None)
        orientation_rad = getattr(region, 'orientation', None)
        orientation_deg = np.degrees(orientation_rad) if orientation_rad is not None else None

        # 质心
        try:
            centroid_y, centroid_x = region.centroid
        except Exception:
            centroid_y, centroid_x = None, None

        # Feret max 近似（最大点对距离）
        max_diameter = compute_max_diameter(region.coords)

        # 外接框与占用率
        try:
            minr, minc, maxr, maxc = region.bbox
            bbox_area = int((maxr - minr) * (maxc - minc))
        except Exception:
            bbox_area = None
        extent = getattr(region, 'extent', None)

        # 凸性度量
        solidity = getattr(region, 'solidity', None)

        # 基于面积的等效直径（API 兼容）
        equivalent_diameter = getattr(region, 'equivalent_diameter_area', None)
        if equivalent_diameter is None:
            equivalent_diameter = getattr(region, 'equivalent_diameter', None)

        # 细胞核覆盖（可选）
        nuclear_area = None
        nuclear_fraction = None
        if nuclei_mask is not None and area > 0:
            try:
                # 将核掩码二值化
                nuc_bin = (nuclei_mask > 0)
                # 当前细胞像素布尔图
                cell_bin = (mask == region.label)
                # 交集像素数
                nuclear_area = int(np.count_nonzero(nuc_bin & cell_bin))
                nuclear_fraction = float(nuclear_area / float(area)) if area > 0 else None
            except Exception:
                nuclear_area = None
                nuclear_fraction = None

        # 核区域 RGB 均值（在原始RGB图像上统计）
        nuclear_mean_R = None
        nuclear_mean_G = None
        nuclear_mean_B = None
        if rgb_img is not None and nuclear_area is not None:
            try:
                rr, cc = region.coords[:, 0], region.coords[:, 1]
                # 细胞内的核像素
                mask_cell = np.zeros(mask.shape, dtype=np.uint8)
                mask_cell[rr, cc] = 1
                nuc_in_cell = (mask_cell & (nuclei_mask > 0)) > 0
                if np.any(nuc_in_cell):
                    if rgb_img.ndim == 3 and rgb_img.shape[2] >= 3:
                        # OpenCV 读取为BGR
                        b = rgb_img[:, :, 0][nuc_in_cell].astype(np.float32)
                        g = rgb_img[:, :, 1][nuc_in_cell].astype(np.float32)
                        r = rgb_img[:, :, 2][nuc_in_cell].astype(np.float32)
                        nuclear_mean_R = float(r.mean())
                        nuclear_mean_G = float(g.mean())
                        nuclear_mean_B = float(b.mean())
            except Exception:
                nuclear_mean_R = nuclear_mean_G = nuclear_mean_B = None

        features = {
            "image": image_filename,
            "label": new_label,
            "area": area,
            "perimeter": perimeter,
            "major_axis_length": major_axis_length,
            "minor_axis_length": minor_axis_length,
            "aspect_ratio": aspect_ratio,
            "eccentricity": eccentricity,
            "orientation_rad": orientation_rad,
            "orientation_deg": orientation_deg,
            "centroid_y": centroid_y,
            "centroid_x": centroid_x,
            "max_diameter": max_diameter,
            "bbox_area": bbox_area,
            "extent": extent,
            "solidity": solidity,
            "equivalent_diameter": equivalent_diameter,
            "nuclear_area": nuclear_area,
            "nuclear_fraction": nuclear_fraction,
            "nuclear_mean_R": nuclear_mean_R,
            "nuclear_mean_G": nuclear_mean_G,
            "nuclear_mean_B": nuclear_mean_B,
        }
        new_label += 1
        features_list.append(features)
    return features_list

def process_single_mask(args):
    """
    处理单个mask文件的函数，用于并行处理
    """
    mask_path, output_csv_folder, nuclei_path, selected_columns, rgb_image_path = args
    filename = os.path.basename(mask_path)
    print(f"Processing {mask_path}...")
    
    mask = imread_unicode(mask_path)
    if mask is None:
        print(f"警告: 无法读取 mask 文件 {mask_path}，请检查文件路径或文件完整性。")
        return

    # 若 mask 为多通道，则转换为单通道
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # 确保 mask 为整数类型
    mask = mask.astype(np.int32)
    
    # 载入已配对的核掩码（若提供）
    nuclei_mask = None
    if nuclei_path:
        nuclei_mask = imread_unicode(nuclei_path)
        if nuclei_mask is not None and nuclei_mask.ndim == 3:
            nuclei_mask = cv2.cvtColor(nuclei_mask, cv2.COLOR_BGR2GRAY)

    # 载入原始RGB图像（可选），用于核区域RGB统计
    rgb_img = None
    if rgb_image_path:
        rgb_img = imread_unicode(rgb_image_path)
    
    features_list = extract_cell_features_from_mask(mask, filename, nuclei_mask=nuclei_mask, rgb_img=rgb_img)
    if not features_list:
        print(f"未在 {filename} 中检测到有效的细胞区域。")
        return

    # 保存 CSV 文件（按需筛选列）
    csv_save_path = os.path.join(output_csv_folder, os.path.splitext(filename)[0] + ".csv")
    df = pd.DataFrame(features_list)
    if selected_columns is not None:
        keep_cols = [c for c in selected_columns if c in df.columns]
        if keep_cols:
            df = df[keep_cols]
    df.to_csv(csv_save_path, index=False)
    print(f"保存 {filename} 的特征到 {csv_save_path}")

def process_masks(masks_folder, output_csv_folder, nuclei_folder=None, selected_columns=None, rgb_folder=None):
    """
    并行处理masks文件夹中的所有mask文件
    """
    if not os.path.exists(output_csv_folder):
        os.makedirs(output_csv_folder)

    import re
    # 仅处理 TIF/TIFF（按用户要求）
    valid_extensions = ('.tif', '.tiff')

    def root_key(name: str) -> str:
        base = os.path.splitext(name)[0]
        # 去除前缀 processed_ 等
        base = re.sub(r'^(processed_)+', '', base, flags=re.IGNORECASE)
        # 去除末尾的 _mask/_masks
        base = re.sub(r'_(mask|masks)$', '', base, flags=re.IGNORECASE)
        return base.lower()

    # 先按根名分组，进行去重选择（仅 .tif/.tiff）
    grouped: dict[str, dict] = {}
    for filename in os.listdir(masks_folder):
        if not filename.lower().endswith(valid_extensions):
            continue
        rk = root_key(filename)
        ext = os.path.splitext(filename)[1].lower()
        current = grouped.get(rk)
        if current is None:
            grouped[rk] = {"filename": filename, "ext": ext}
        else:
            # 如同名存在多种扩展，仅保留首次扫描到的 .tif/.tiff（两者等价处理）
            def rank(e: str) -> int:
                return 0 if e in ('.tif', '.tiff') else 1
            if rank(ext) < rank(current["ext"]):
                grouped[rk] = {"filename": filename, "ext": ext}

    # 转为处理列表（同时为每个细胞掩码匹配核掩码路径）
    selected = list(grouped.values())

    nuclei_index: dict[str, str] = {}
    if nuclei_folder and os.path.isdir(nuclei_folder):
        # 构建核索引：按规范化根名并按扩展优先级去重（仅 .tif/.tiff）
        nuc_grouped: dict[str, dict] = {}
        for fn in os.listdir(nuclei_folder):
            if not fn.lower().endswith(valid_extensions):
                continue
            rk = root_key(fn)
            ext = os.path.splitext(fn)[1].lower()
            cur = nuc_grouped.get(rk)
            def rank(e: str) -> int:
                return 0 if e in ('.tif', '.tiff') else 1
            if cur is None or rank(ext) < rank(cur["ext"]):
                nuc_grouped[rk] = {"filename": fn, "ext": ext}
        nuclei_index = {rk: os.path.join(nuclei_folder, info["filename"]) for rk, info in nuc_grouped.items()}

    # 构建原图索引（可选）
    rgb_index: dict[str, str] = {}
    if rgb_folder and os.path.isdir(rgb_folder):
        for fn in os.listdir(rgb_folder):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
                continue
            rk = root_key(fn)
            rgb_index[rk] = os.path.join(rgb_folder, fn)

    mask_files = []
    for item in selected:
        mpath = os.path.join(masks_folder, item["filename"])
        rk = root_key(item["filename"])  # 规范化根名匹配
        npath = nuclei_index.get(rk)
        rpath = rgb_index.get(rk)
        mask_files.append((mpath, output_csv_folder, npath, selected_columns, rpath))

    skipped = len([1 for _ in os.listdir(masks_folder) if _.lower().endswith(valid_extensions)]) - len(mask_files)
    print(f"在 {masks_folder} 发现 {len(mask_files)} 份掩码（去重后），跳过 {skipped} 份重复（基于根名，仅保留 .tif/.tiff）。")
    if nuclei_folder and os.path.isdir(nuclei_folder):
        print(f"已从 {nuclei_folder} 匹配到 {sum(1 for t in mask_files if len(t) >= 3 and t[2])} 份核掩码（按根名匹配，忽略前缀 processed_ / 后缀 _mask/_masks，仅 .tif/.tiff）。")
    
    if not mask_files:
        print(f"在 {masks_folder} 中没有找到有效的掩码文件（支持: .tif/.tiff）。")
        return

    # 使用进程池并行处理
    num_processes = max(1, cpu_count() - 1)  # 保留一个CPU核心
    print(f"使用 {num_processes} 个进程进行并行处理")
    # 预热一次，便于尽早暴露单文件问题
    if mask_files:
        try:
            process_single_mask(mask_files[0])
        except Exception as e:
            print(f"预热处理首个文件失败: {e}")
    
    with Pool(processes=num_processes) as pool:
        for _ in pool.imap_unordered(process_single_mask, mask_files, chunksize=2):
            pass

def process_all_day_folders(data_root, nuclei_subfolder_name='nuclei', selected_columns=None, rgb_subfolder='images'):
    """
    处理所有DAY文件夹中的masks
    """
    print(f"开始扫描目录: {data_root}")
    
    # 遍历所有 Day* / DAY* 文件夹（大小写不敏感）
    entries = [d for d in sorted(os.listdir(data_root))
               if os.path.isdir(os.path.join(data_root, d)) and d.strip().lower().startswith('day')]
    if not entries:
        print("未发现以 Day*/DAY* 命名的子目录。")
        return

    for day_folder in entries:
        day_path = os.path.join(data_root, day_folder)
        masks_folder = os.path.join(day_path, 'masks')
        features_folder = os.path.join(day_path, 'features')
        nuclei_folder = os.path.join(day_path, nuclei_subfolder_name)
        rgb_folder = os.path.join(day_path, rgb_subfolder) if rgb_subfolder else None

        if os.path.exists(masks_folder):
            print(f"\n=== 处理 {day_folder} ===")
            print(f"masks文件夹: {masks_folder}")
            print(f"features文件夹: {features_folder}")
            if os.path.exists(nuclei_folder):
                print(f"nuclei文件夹: {nuclei_folder}")
            else:
                print("nuclei文件夹: 未提供（将仅输出细胞特征）")

            process_masks(
                masks_folder,
                features_folder,
                nuclei_folder if os.path.exists(nuclei_folder) else None,
                selected_columns,
                rgb_folder if (rgb_folder and os.path.exists(rgb_folder)) else None,
            )
        else:
            print(f"跳过 {day_folder}: masks文件夹不存在")

def main():
    parser = argparse.ArgumentParser(description="从 masks/* 提取细胞/细胞核特征并保存到 features/*")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据根目录（包含各 DAY* 子文件夹）。若不提供则使用相对路径 ../data",
    )
    parser.add_argument(
        "--nuclei-subfolder",
        type=str,
        default='nuclei',
        help="核掩码所在子文件夹名称（与每个DAY同级，默认 'nuclei'）。该文件夹中核掩码需与细胞掩码同名。",
    )
    parser.add_argument(
        "--rgb-subfolder",
        type=str,
        default='images',
        help="用于统计核RGB的原图所在子文件夹（默认 'images'；若设为空则不统计RGB）",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default='all',
        help="选择输出列：'all'（默认）或 'minimal' 或 以逗号分隔的列名列表",
    )
    args = parser.parse_args()

    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    data_root = args.data_root or os.environ.get('DATA_ROOT') or default_root

    # 解析列选择
    selected_columns = None
    if args.columns and args.columns.lower() != 'all':
        if args.columns.lower() == 'minimal':
            selected_columns = [
                'image','label','area','perimeter',
                'major_axis_length','minor_axis_length','aspect_ratio',
                'eccentricity','orientation_deg','centroid_x','centroid_y',
                'nuclear_area','nuclear_fraction','nuclear_mean_R','nuclear_mean_G','nuclear_mean_B'
            ]
        else:
            selected_columns = [s.strip() for s in args.columns.split(',') if s.strip()]

    if selected_columns:
        print("仅输出列: " + ", ".join(selected_columns))
    print(f"开始提取特征... 数据根目录: {data_root}")
    process_all_day_folders(
        data_root,
        nuclei_subfolder_name=args.nuclei_subfolder,
        selected_columns=selected_columns,
        rgb_subfolder=args.rgb_subfolder,
    )
    print("\n所有特征提取完成！")

if __name__ == '__main__':
    main()



