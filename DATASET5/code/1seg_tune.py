#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import time
from itertools import product
from typing import List

import cv2
import numpy as np
from cellpose import models, io


def imread_unicode(file_path):
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading image: {e}")
        return None


def normalize99(img: np.ndarray) -> np.ndarray:
    X = img.astype(np.float32)
    p1 = float(np.percentile(X, 1))
    p99 = float(np.percentile(X, 99))
    X = (X - p1) / (1e-10 + (p99 - p1))
    X = np.clip(X, 0.0, 1.0)
    return X


def image_resize_max(img: np.ndarray, max_resize: int = 1000) -> np.ndarray:
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
    if background_bgr.ndim == 2:
        background_bgr = cv2.cvtColor(background_bgr, cv2.COLOR_GRAY2BGR)
    overlay = background_bgr.copy()
    contours, _ = cv2.findContours(masks.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    line_type = cv2.LINE_AA if antialias else cv2.LINE_8
    cv2.drawContours(overlay, contours, contourIdx=-1, color=(0, 0, 255), thickness=max(thickness, 2), lineType=line_type)
    cv2.drawContours(overlay, contours, contourIdx=-1, color=(0, 255, 255), thickness=1, lineType=line_type)
    return overlay


def imread_mask_unicode(file_path):
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception as e:
        print(f"Error reading mask: {e}")
        return None


def compute_binary_metrics(gt: np.ndarray, pred: np.ndarray):
    gt_bin = (gt > 0).astype(np.uint8)
    pred_bin = (pred > 0).astype(np.uint8)
    tp = int(np.sum((gt_bin == 1) & (pred_bin == 1)))
    fp = int(np.sum((gt_bin == 0) & (pred_bin == 1)))
    fn = int(np.sum((gt_bin == 1) & (pred_bin == 0)))
    tn = int(np.sum((gt_bin == 0) & (pred_bin == 0)))
    eps = 1e-10
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        'precision': precision,
        'recall': recall,
        'dice': dice,
        'iou': iou,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
    }


def parse_list_int(s: str) -> List[int]:
    return [int(x) for x in s.split(',') if x.strip() != '']


def parse_list_float(s: str) -> List[float]:
    return [float(x) for x in s.split(',') if x.strip() != '']


def sanitize_tag(value: float | int) -> str:
    text = str(value)
    return text.replace('.', 'p').replace('-', 'm')


def run_grid_on_folder(model: models.CellposeModel,
                       img_folder: str,
                       out_folder_root: str,
                       diameter_values: List[float],
                       resize_max: int,
                       niter_values: List[int],
                       flow_values: List[float],
                       cellprob_values: List[float],
                       min_size_values: List[int],
                       channels: List[int],
                       invert: bool,
                       save_outlines: bool = True,
                       limit: int | None = None,
                       img_paths_override: List[str] | None = None,
                       gt_mask_path: str | None = None,
                       results_csv_path: str | None = None):
    if not os.path.exists(img_folder):
        print(f"[跳过] 未找到图片目录: {img_folder}")
        return

    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    if img_paths_override is not None:
        img_paths = img_paths_override
    else:
        img_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.lower().endswith(valid_extensions)]
    if limit is not None:
        img_paths = img_paths[:max(0, int(limit))]
    if not img_paths:
        print(f"[跳过] {img_folder} 中没有有效图片")
        return

    print(f"[INFO] 数据: {img_folder} | 图片数: {len(img_paths)}")

    for diam, niter, flow_thr, cellprob_thr, minsz in product(diameter_values, niter_values, flow_values, cellprob_values, min_size_values):
        dtag = f"d{sanitize_tag(diam)}" if float(diam) > 0 else "dauto"
        tag = f"{dtag}_n{sanitize_tag(niter)}_f{sanitize_tag(flow_thr)}_c{sanitize_tag(cellprob_thr)}_ms{sanitize_tag(minsz)}"
        out_folder = os.path.join(out_folder_root, f"tune_{tag}")
        os.makedirs(out_folder, exist_ok=True)
        print(f"\n[参数组合] diameter={diam}, niter={niter}, flow={flow_thr}, cellprob={cellprob_thr}, min_size={minsz}, channels={channels}")
        print(f"输出目录: {out_folder}")

        start = time.time()
        completed = 0
        for idx, img_path in enumerate(img_paths, start=1):
            filename = os.path.basename(img_path)
            print(f"[{idx}/{len(img_paths)}] 处理: {filename}")
            try:
                img = imread_unicode(img_path)
                if img is None:
                    print(f"[{idx}/{len(img_paths)}] 警告: 无法读取，跳过")
                    continue

                h0, w0 = img.shape[:2]
                img_pre = (255 - img) if invert else img
                img_proc = image_resize_max(img_pre, max_resize=resize_max) if resize_max else img_pre
                img_proc = normalize99(img_proc)
                print(f"[{idx}/{len(img_paths)}] {filename}: 已应用 normalize99{(' + resize' if resize_max else '')}{(' + invert' if invert else '')}")

                result = model.eval(
                    img_proc,
                    diameter=(float(diam) if float(diam) > 0 else None),
                    niter=int(niter),
                    flow_threshold=float(flow_thr),
                    cellprob_threshold=float(cellprob_thr),
                    min_size=int(minsz),
                    channels=channels[:2] if len(channels) >= 2 else [0, 0]
                )
                masks = result[0] if isinstance(result, tuple) else result
                roi_count = len(np.unique(masks)) - 1 if len(np.unique(masks)) > 1 else 0
                print(f"[{idx}/{len(img_paths)}] {filename}: 检测到 {roi_count} 个ROI")

                if masks.shape[0] != h0 or masks.shape[1] != w0:
                    masks = cv2.resize(masks.astype('uint16'), (w0, h0), interpolation=cv2.INTER_NEAREST).astype('uint16')

                base = os.path.splitext(filename)[0]
                mask_png = os.path.join(out_folder, f"{base}_mask.png")
                mask_tif = os.path.join(out_folder, f"{base}_masks.tif")
                cv2.imwrite(mask_png, ((masks > 0) * 255).astype(np.uint8))
                try:
                    io.imsave(mask_tif, masks.astype('uint16'))
                except Exception:
                    pass

                if save_outlines:
                    try:
                        outlines_img = draw_outlines_overlay(img, masks, thickness=2, antialias=True)
                        outlines_png = os.path.join(out_folder, f"{base}_outlines.png")
                        cv2.imwrite(outlines_png, outlines_img)
                    except Exception as e:
                        print(f"[{idx}/{len(img_paths)}] 绘制outlines失败: {e}")

                # 评估（若提供 GT 且是单图场景）
                if gt_mask_path is not None and (img_paths_override is not None and len(img_paths_override) == 1):
                    try:
                        gt = imread_mask_unicode(gt_mask_path)
                        if gt is not None:
                            if gt.shape[:2] != masks.shape[:2]:
                                gt = cv2.resize(gt, (masks.shape[1], masks.shape[0]), interpolation=cv2.INTER_NEAREST)
                            metrics = compute_binary_metrics(gt, (masks > 0).astype(np.uint8))
                            print(f"[评估] IoU={metrics['iou']:.4f} | Dice={metrics['dice']:.4f} | P={metrics['precision']:.4f} | R={metrics['recall']:.4f} | Acc={metrics['accuracy']:.4f}")
                            if results_csv_path:
                                header_needed = not os.path.exists(results_csv_path)
                                with open(results_csv_path, 'a', encoding='utf-8') as f:
                                    if header_needed:
                                        f.write('image,diameter,niter,flow,cellprob,min_size,channels,iou,dice,precision,recall,accuracy\n')
                                    ch_str = f"{channels[0]}:{channels[1]}" if len(channels) >= 2 else '0:0'
                                    f.write(f"{os.path.basename(img_path)},{diam},{niter},{flow_thr},{cellprob_thr},{minsz},{ch_str},{metrics['iou']:.6f},{metrics['dice']:.6f},{metrics['precision']:.6f},{metrics['recall']:.6f},{metrics['accuracy']:.6f}\n")
                    except Exception as e:
                        print(f"[评估失败] {e}")

            except Exception as e:
                print(f"[{idx}/{len(img_paths)}] 分割出错: {e}")

            completed += 1
            elapsed = time.time() - start
            avg = elapsed / completed if completed else 0
            eta = (len(img_paths) - completed) * avg
            print(f"[进度] {completed}/{len(img_paths)} ({completed/len(img_paths)*100:.1f}%) | 用时 {elapsed/60:.1f} 分 | 预计剩余 {eta/60:.1f} 分")


def main():
    parser = argparse.ArgumentParser(description="顺序网格调参：对单个文件或文件夹进行多组参数分割")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", type=str, help="图片文件夹路径，仅处理该目录下的图像")
    src.add_argument("--image", type=str, help="单张图片路径，仅处理该图片")
    parser.add_argument("--output", type=str, default=None, help="输出根目录（默认: 在 images 下创建 tune_results）")
    parser.add_argument("--model-type", type=str, choices=["cpsam", "cyto3"], default="cpsam", help="选择分割模型类型")
    parser.add_argument("--invert", action="store_true", help="对明场图像进行反相后再分割")
    parser.add_argument("--gt", type=str, default=None, help="(单图) 手工标注mask路径，用于计算IoU/Dice/准确率等")

    parser.add_argument("--diameter", type=str, default="0", help="细胞直径像素，逗号分隔；0 表示自动")
    parser.add_argument("--resize-max", type=int, default=1000, help="最大边缩放尺寸，0 表示不缩放")
    parser.add_argument("--niter", type=str, default="250", help="逗号分隔，如: 200,250,300")
    parser.add_argument("--flow-threshold", type=str, default="0.4", help="逗号分隔，如: 0.4,0.5")
    parser.add_argument("--cellprob-threshold", type=str, default="0.0", help="逗号分隔，如: -0.2,0.0,0.2")
    parser.add_argument("--min-size", type=str, default="10", help="目标最小像素面积，逗号分隔；默认 10")
    parser.add_argument("--channels", type=str, default="0,0", help="Cellpose 通道设置，形如: 0,0 或 2,3")

    parser.add_argument("--no-outlines", action="store_true", help="不保存outlines可视化")
    parser.add_argument("--limit", type=int, default=None, help="每个文件夹最多处理前N张图，用于快速测试")
    parser.add_argument("--cpu", action="store_true", help="强制CPU推理")

    args = parser.parse_args()

    diameter_values = parse_list_float(args.diameter)
    niter_values = parse_list_int(args.niter)
    flow_values = parse_list_float(args.flow_threshold)
    cellprob_values = parse_list_float(args.cellprob_threshold)
    min_size_values = parse_list_int(args.min_size)
    channels = [int(x) for x in args.channels.split(',') if x.strip() != '']
    resize_max = int(args.resize_max) if int(args.resize_max) > 0 else 0
    save_outlines = not args.no_outlines

    if args.image:
        image_path = os.path.abspath(args.image)
        images_dir = os.path.dirname(image_path)
        out_root = args.output or os.path.join(images_dir, 'tune_results')
        img_paths_override = [image_path]
        print(f"image: {image_path}")
        results_csv_path = os.path.join(out_root, 'tune_metrics.csv') if args.gt else None
    else:
        images_dir = os.path.abspath(args.images)
        out_root = args.output or os.path.join(images_dir, 'tune_results')
        img_paths_override = None
        print(f"images: {images_dir}")
        results_csv_path = None
    os.makedirs(out_root, exist_ok=True)
    print(f"网格: diameter={diameter_values}, niter={niter_values}, flow={flow_values}, cellprob={cellprob_values}, min_size={min_size_values}")
    print(f"channels={channels}")
    print(f"resize_max={resize_max} | outlines={save_outlines} | limit={args.limit} | CPU={args.cpu}")

    print("正在初始化模型...")
    try:
        model = models.CellposeModel(model_type=args.model_type, gpu=(not args.cpu))
    except Exception as e:
        print(f"[提示] GPU 初始化失败，将回退 CPU: {e}")
        model = models.CellposeModel(model_type=args.model_type, gpu=False)

    run_grid_on_folder(
        model=model,
        img_folder=images_dir,
        out_folder_root=out_root,
        diameter_values=diameter_values,
        resize_max=resize_max,
        niter_values=niter_values,
        flow_values=flow_values,
        cellprob_values=cellprob_values,
        min_size_values=min_size_values,
        channels=channels,
        invert=args.invert,
        save_outlines=save_outlines,
        limit=args.limit,
        img_paths_override=img_paths_override,
        gt_mask_path=(os.path.abspath(args.gt) if args.gt else None),
        results_csv_path=results_csv_path,
    )

    del model

    print("\n全部参数组合处理完成！")


if __name__ == '__main__':
    main()


