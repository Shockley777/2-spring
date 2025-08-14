#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import cv2
import numpy as np
from cellpose import models, io
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional
import importlib

# 动态导入 huggingface_hub，避免环境未安装时报未解析导入
try:
    huggingface_hub = importlib.import_module('huggingface_hub')
    hf_hub_download = getattr(huggingface_hub, 'hf_hub_download', None)
except Exception:
    hf_hub_download = None

def preprocess_nuclei_blue_invert_bg(img_bgr: np.ndarray, rolling_radius: int = 10) -> np.ndarray:
    """
    近似复现 FIJI 流程：Split Channels -> Blue -> Invert -> Subtract Background (rolling=radius)
    - 输入: BGR 或灰度图
    - 输出: 单通道 uint8 图像
    """
    if img_bgr.ndim == 3:
        blue = img_bgr[:, :, 0]
    else:
        blue = img_bgr

    # Invert
    blue_inv = (255 - blue).astype(np.uint8)

    # Rolling ball 背景扣除（用形态学开运算近似）
    k = max(1, int(rolling_radius) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(blue_inv, cv2.MORPH_OPEN, kernel)
    sub = cv2.subtract(blue_inv, background)
    return sub

def imread_unicode(file_path):
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading image: {e}")
        return None


def normalize99(img: np.ndarray) -> np.ndarray:
    """按1-99百分位归一化到[0,1]并裁剪，匹配Huggingface中CellposeSAM预处理。"""
    X = img.astype(np.float32)
    p1 = float(np.percentile(X, 1))
    p99 = float(np.percentile(X, 99))
    X = (X - p1) / (1e-10 + (p99 - p1))
    X = np.clip(X, 0.0, 1.0)
    return X


def image_resize_max(img: np.ndarray, max_resize: int = 1000) -> np.ndarray:
    """将图像按较大边等比缩放到不超过 max_resize，匹配 app.py 的 image_resize。"""
    ny, nx = img.shape[:2]
    if max(ny, nx) <= max_resize:
        return img
    if ny > nx:
        nx = int(nx / ny * max_resize)
        ny = max_resize
    else:
        ny = int(ny / nx * max_resize)
        nx = max_resize
    resized = cv2.resize(img, (nx, ny))
    return resized.astype(img.dtype)


def draw_outlines_overlay(background_bgr: np.ndarray, masks: np.ndarray, thickness: int = 2, antialias: bool = True) -> np.ndarray:
    """在原图上绘制轮廓，返回BGR图像（支持加粗与抗锯齿）。"""
    if background_bgr.ndim == 2:
        background_bgr = cv2.cvtColor(background_bgr, cv2.COLOR_GRAY2BGR)
    overlay = background_bgr.copy()
    try:
        contours, _ = cv2.findContours(masks.astype(np.int32), mode=cv2.RETR_FLOODFILL, method=cv2.CHAIN_APPROX_SIMPLE)
    except Exception:
        contours, _ = cv2.findContours(masks.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    line_type = cv2.LINE_AA if antialias else cv2.LINE_8
    # 双层描边提升观感：先粗红后细黄
    cv2.drawContours(overlay, contours, contourIdx=-1, color=(0, 0, 255), thickness=max(thickness, 2), lineType=line_type)
    cv2.drawContours(overlay, contours, contourIdx=-1, color=(0, 255, 255), thickness=1, lineType=line_type)
    return overlay



def load_hf_app_model(use_gpu: bool = True) -> Optional[models.CellposeModel]:
    """从 Huggingface 下载并加载 app.py 使用的 cpsam 权重。加载失败返回 None。"""
    if hf_hub_download is None:
        print("[提示] 未安装 huggingface_hub，将回退为内置 model_type='cpsam'。", flush=True)
        return None
    try:
        fpath = hf_hub_download(repo_id="mouseland/cellpose-sam", filename="cpsam")
        print(f"[HF] 权重已下载: {fpath}", flush=True)
        return models.CellposeModel(gpu=use_gpu, pretrained_model=fpath)
    except Exception as e:
        print(f"[HF] 加载 Huggingface 权重失败，将回退为内置模型: {e}", flush=True)
        return None


def process_single_image(args):
    img_path, output_folder, use_gpu, current_index, total_count = args
    filename = os.path.basename(img_path)
    
    print(f"[{current_index}/{total_count}] 开始处理: {filename}", flush=True)
    
    try:
        img = imread_unicode(img_path)
        if img is None:
            print(f"[{current_index}/{total_count}] 警告: 无法读取图片 {filename}, 跳过", flush=True)
            return

        # 归一化到[0,1]，与Huggingface一致
        img = normalize99(img)
        print(f"[{current_index}/{total_count}] {filename}: 已应用 normalize99", flush=True)

        print(f"[{current_index}/{total_count}] {filename}: 正在加载模型...", flush=True)
        # 每个进程独立初始化模型
        model = models.CellposeModel(model_type='cpsam', gpu=use_gpu)
        
        print(f"[{current_index}/{total_count}] {filename}: 开始分割...", flush=True)
        result = model.eval(
            img,
            diameter=56,
            flow_threshold=0.5,
            cellprob_threshold=0.0,
            min_size=10,
            niter=280
        )
        
        if isinstance(result, tuple):
            masks = result[0]
        else:
            masks = result
        
        # 不应用任何后处理，保持原始分割结果
        roi_count = len(np.unique(masks)) - 1 if len(np.unique(masks)) > 1 else 0
        print(f"[{current_index}/{total_count}] {filename}: 检测到 {roi_count} 个ROI", flush=True)

        # 保存PNG掩码
        mask_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_mask.png")
        cv2.imwrite(mask_filename, ((masks > 0) * 255).astype(np.uint8))
        print(f"[{current_index}/{total_count}] {filename}: 掩码已保存: {os.path.basename(mask_filename)}", flush=True)
        
        # 清理模型内存
        del model
        
    except Exception as e:
        print(f"[{current_index}/{total_count}] {filename}: 分割过程中出错: {e}", flush=True)
        import traceback
        traceback.print_exc()

def segment_folder(input_folder, output_folder, use_gpu=True, num_workers=6,
                   use_hf_app: bool = True,
                   hf_max_resize: int = 1000,
                   hf_niter: int = 250,
                   hf_flow_threshold: float = 0.4,
                    hf_cellprob_threshold: float = 0.0,
                    mode: str = 'cells',
                    nuclei_rolling_radius: int = 10,
                    nuclei_preprocessed: bool = False):
    """
    多进程分割单个文件夹中的图像
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    img_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
                 if f.lower().endswith(valid_extensions)]
    
    if not img_paths:
        print(f"在 {input_folder} 中没有找到有效图片", flush=True)
        return 0
    
    # GPU 并发通常会导致卡顿或显存不足，强制将并发降为 1
    if use_gpu and num_workers > 1:
        print(f"检测到 GPU 推理，出于稳定性将并发进程数从 {num_workers} 降为 1（避免显存竞争导致卡住）", flush=True)
        num_workers = 1

    if use_gpu:
        if use_hf_app:
            print(f"处理文件夹 {input_folder}: 共{len(img_paths)}张图片，GPU 单模型顺序处理（HF app 风格）。", flush=True)
        else:
            print(f"处理文件夹 {input_folder}: 共{len(img_paths)}张图片，GPU 单模型顺序处理。", flush=True)
    else:
        print(f"处理文件夹 {input_folder}: 共{len(img_paths)}张图片，使用{num_workers}进程并行处理。", flush=True)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    try:
        start_time = time.time()
        completed = 0

        if use_gpu:
            # 加载一次模型，顺序处理（优先使用 Huggingface 权重）
            print("正在初始化单实例模型...", flush=True)
            model = None
            if use_hf_app:
                model = load_hf_app_model(use_gpu=True)
            if model is None:
                model = models.CellposeModel(model_type='cpsam', gpu=True)

            for idx, img_path in enumerate(img_paths, start=1):
                filename = os.path.basename(img_path)
                print(f"[{idx}/{len(img_paths)}] 开始处理: {filename}", flush=True)
                try:
                    img = imread_unicode(img_path)
                    if img is None:
                        print(f"[{idx}/{len(img_paths)}] 警告: 无法读取图片 {filename}, 跳过", flush=True)
                        continue

                    # 模式预处理
                    if mode == 'nuclei' and not nuclei_preprocessed:
                        img = preprocess_nuclei_blue_invert_bg(img, rolling_radius=nuclei_rolling_radius)

                    # HF app 风格：先缩放（若启用）再归一化
                    original_h, original_w = img.shape[:2]
                    img_proc = image_resize_max(img, max_resize=hf_max_resize) if use_hf_app else img
                    img_proc = normalize99(img_proc)
                    print(f"[{idx}/{len(img_paths)}] {filename}: 已应用 normalize99{(' + resize' if use_hf_app else '')}", flush=True)

                    # 推理参数：核使用与细胞相同的默认参数；其余情况下可走 HF 风格或默认
                    if mode == 'nuclei':
                        result = model.eval(
                            img_proc,
                            diameter=56,
                            flow_threshold=0.5,
                            cellprob_threshold=0.0,
                            min_size=10,
                            niter=280
                        )
                    else:
                        if use_hf_app:
                            result = model.eval(
                                img_proc,
                                niter=hf_niter,
                                flow_threshold=hf_flow_threshold,
                                cellprob_threshold=hf_cellprob_threshold
                            )
                        else:
                            result = model.eval(
                                img_proc,
                                diameter=56,
                                flow_threshold=0.5,
                                cellprob_threshold=0.0,
                                min_size=10,
                                niter=280
                            )

    
                    # 解包结果：仅关心 masks
                    if isinstance(result, tuple):
                        masks = result[0]
                    else:
                        masks = result
                    roi_count = len(np.unique(masks)) - 1 if len(np.unique(masks)) > 1 else 0
                    print(f"[{idx}/{len(img_paths)}] {filename}: 检测到 {roi_count} 个ROI", flush=True)

                    # 如果做过缩放，按最近邻缩放回原始尺寸
                    if use_hf_app and (masks.shape[0] != original_h or masks.shape[1] != original_w):
                        masks = cv2.resize(masks.astype('uint16'), (original_w, original_h), interpolation=cv2.INTER_NEAREST).astype('uint16')

                    # 保存：同时保存 png 二值与 tif（与 app 一致）
                    base = os.path.splitext(filename)[0]
                    mask_png = os.path.join(output_folder, f"{base}_mask.png")
                    mask_tif = os.path.join(output_folder, f"{base}_masks.tif")
                    cv2.imwrite(mask_png, ((masks > 0) * 255).astype(np.uint8))
                    try:
                        io.imsave(mask_tif, masks.astype('uint16'))
                    except Exception:
                        pass
                    print(f"[{idx}/{len(img_paths)}] {filename}: 掩码已保存: {os.path.basename(mask_png)}", flush=True)

                    # 生成并保存 outlines 可视化
                    try:
                        outlines_img = draw_outlines_overlay(img, masks, thickness=2, antialias=True)
                        outlines_png = os.path.join(output_folder, f"{base}_outlines.png")
                        cv2.imwrite(outlines_png, outlines_img)
                    except Exception as e:
                        print(f"[{idx}/{len(img_paths)}] {filename}: 绘制outlines失败: {e}", flush=True)


                except Exception as e:
                    print(f"[{idx}/{len(img_paths)}] {filename}: 分割过程中出错: {e}", flush=True)

                completed += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / completed if completed else 0.0
                remaining = len(img_paths) - completed
                eta = remaining * avg_time
                print(f"[进度] {completed}/{len(img_paths)} 完成 ({completed/len(img_paths)*100:.1f}%) | 耗时 {elapsed/60:.1f} 分钟 | 预计剩余 {eta/60:.1f} 分钟", flush=True)

            # 释放模型
            del model
        else:
            # CPU 并行：每进程各自加载模型
            args_list = [(img_path, output_folder, False, i+1, len(img_paths)) 
                         for i, img_path in enumerate(img_paths)]

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process_single_image, args): args for args in args_list}
                for future in as_completed(futures):
                    completed += 1
                    args = futures[future]
                    current_index = args[3]
                    total_count = args[4]

                    try:
                        future.result()
                        elapsed = time.time() - start_time
                        avg_time = elapsed / completed if completed else 0.0
                        remaining = total_count - completed
                        eta = remaining * avg_time
                        print(f"[进度] {completed}/{total_count} 完成 ({completed/total_count*100:.1f}%) | 耗时 {elapsed/60:.1f} 分钟 | 预计剩余 {eta/60:.1f} 分钟", flush=True)
                    except Exception as e:
                        print(f"[错误] 图片 {current_index}/{total_count} 处理失败: {e}", flush=True)

                    if completed % 10 == 0 or completed == total_count:
                        print(f"[进度更新] 已完成 {completed}/{total_count} 张图片 ({completed/total_count*100:.1f}%)", flush=True)
                    
    except Exception as e:
        print(f"并行处理过程中出错: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    print(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"文件夹 {input_folder} 处理完成！", flush=True)
    return completed

def process_all_day_folders(data_root, mode: str = 'cells', nuclei_rolling_radius: int = 10,
                            input_subdir: str | None = None,
                            output_subdir: str | None = None):
    """
    处理所有DAY文件夹
    """
    # 选择输入/输出子目录
    in_sub = input_subdir if input_subdir else ('images' if mode == 'cells' else 'processed')
    out_sub = output_subdir if output_subdir else ('masks' if mode == 'cells' else 'nuclei')
    nuclei_preprocessed = (mode == 'nuclei' and in_sub.lower() == 'processed')

    # 预扫描总图片数用于整体进度
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    total_images_overall = 0
    # 兼容 Dataset7 的 "Day X" 命名（大小写不敏感）
    day_folders = [
        d for d in sorted(os.listdir(data_root))
        if os.path.isdir(os.path.join(data_root, d)) and d.strip().lower().startswith('day')
    ]
    for day_folder in day_folders:
        images_folder = os.path.join(data_root, day_folder, in_sub)
        if os.path.exists(images_folder):
            total_images_overall += sum(1 for f in os.listdir(images_folder) if f.lower().endswith(valid_extensions))

    processed_overall = 0
    print(f"将处理 {len(day_folders)} 个 day* 文件夹，共计 {total_images_overall} 张图片。", flush=True)

    # 遍历所有DAY文件夹
    for day_folder in day_folders:
        day_path = os.path.join(data_root, day_folder)
        if os.path.isdir(day_path):
            print(f"\n=== 处理 {day_folder} ===", flush=True)
            
            # 设置输入和输出路径（可定制子目录）
            images_folder = os.path.join(day_path, in_sub)
            masks_folder = os.path.join(day_path, out_sub)
            
            # 检查images文件夹是否存在
            if not os.path.exists(images_folder):
                print(f"警告: {day_folder} 中没有找到 {in_sub} 文件夹，跳过", flush=True)
                continue
            
            # 执行分割
            done = segment_folder(
                images_folder, masks_folder,
                use_gpu=True, num_workers=6,
                mode=mode,
                nuclei_rolling_radius=nuclei_rolling_radius,
                nuclei_preprocessed=nuclei_preprocessed,
            )
            processed_overall += done
            if total_images_overall > 0:
                print(f"[总体进度] 已完成 {processed_overall}/{total_images_overall} ({processed_overall/total_images_overall*100:.1f}%)", flush=True)

def main():
    parser = argparse.ArgumentParser(description="分割 DAY*/<input-subdir> 并生成 DAY*/<output-subdir> 掩码")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据根目录（包含各 DAY* 子文件夹）。若不提供则使用相对路径 ../data",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cells", "nuclei"],
        default="cells",
        help="分割模式：cells=细胞（默认），nuclei=细胞核（会进行蓝通道+反色+背景扣除预处理）",
    )
    parser.add_argument(
        "--nuclei-rolling-radius",
        type=int,
        default=10,
        help="核预处理 rolling ball 半径（像素），默认 10",
    )
    parser.add_argument(
        "--input-subdir",
        type=str,
        default=None,
        help="输入子目录名称（默认：cells→images；nuclei→processed）",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help="输出子目录名称（默认：cells→masks；nuclei→nuclei）",
    )
    args = parser.parse_args()

    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    data_root = args.data_root or os.environ.get('DATA_ROOT') or default_root

    print(f"开始处理数据... 数据根目录: {data_root} | 模式: {args.mode}", flush=True)
    process_all_day_folders(
        data_root,
        mode=args.mode,
        nuclei_rolling_radius=args.nuclei_rolling_radius,
        input_subdir=args.input_subdir,
        output_subdir=args.output_subdir,
    )
    print("\n所有图片处理完成！", flush=True)

if __name__ == '__main__':
    main()
