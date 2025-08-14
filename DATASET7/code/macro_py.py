#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速替代 ImageJ 宏 macro.ijm 的 Python 实现（更快、可批处理）
流程与 macro.ijm 等价：
- 读取 JPG
- 拆分通道并取蓝通道
- 反相（Invert）
- 背景扣除（可选三种后端）
  - morph: 形态学开运算近似 Rolling Ball（默认，更快）
  - skimage: 使用 scikit-image 的 rolling_ball（与 ImageJ 更接近）
  - imagej: 通过 PyImageJ 调用 Fiji 的 Subtract Background...（与 ImageJ 一致）
- 保存单通道 8-bit TIFF，文件名为 processed_*.tif

使用示例（Windows）：
  python DATASET5\code\macro_py.py -i D:\input_images -o D:\output_masks -r 10 --backend imagej --workers auto

参数说明：
- -i/--input:   输入文件夹（仅处理顶层 .jpg，与 ImageJ 宏一致）
- -o/--output:  输出文件夹（自动创建）
- -r/--rolling: 滚动半径（像素），等价于 ImageJ 的 rolling=10（默认10）
- --backend:    背景扣除后端，可选 morph|skimage|imagej，默认 morph
- --no-sliding: 后端为 imagej 时，禁用 sliding paraboloid（默认启用）
- --workers:    并行线程数，auto=CPU-1；1 表示单线程
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import tempfile

# 懒加载的 ImageJ 上下文（仅当 --backend imagej 时使用）
_IJ_CTX = None
_IJ_CLASS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast replacement of ImageJ macro.ijm (blue→invert→background subtraction)")
    parser.add_argument("-i", "--input", required=True, help="Input directory (only top-level .jpg files will be processed)")
    parser.add_argument("-o", "--output", required=True, help="Output directory for processed TIFF files")
    parser.add_argument("-r", "--rolling", type=int, default=10, help="Rolling radius (pixels). Equivalent to ImageJ rolling=10")
    parser.add_argument("--backend", choices=["morph", "skimage", "imagej"], default="morph", help="Background subtraction backend: morph|skimage|imagej")
    parser.add_argument("--no-sliding", action="store_true", help="Disable 'sliding' paraboloid (only for --backend imagej). Default: sliding enabled")
    parser.add_argument("--workers", default="auto", help="Number of threads to use (int or 'auto'). Default: auto")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards to split the workload")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index for this process (0-based)")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_jpgs(input_dir: Path) -> list[Path]:
    # 与 ImageJ getFileList 行为一致：仅顶层，不递归
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"])


def subtract_background_morphology(img_u8: np.ndarray, radius: int) -> np.ndarray:
    """用形态学开运算近似 ImageJ 的 Rolling Ball 背景扣除。
    背景 = 开运算(img, radius)，结果 = img - 背景。
    """
    size = max(1, 2 * int(radius) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    background = cv2.morphologyEx(img_u8, cv2.MORPH_OPEN, kernel)
    corrected = cv2.subtract(img_u8, background)
    return corrected


def subtract_background_skimage(img_u8: np.ndarray, radius: int) -> np.ndarray:
    """使用 scikit-image rolling_ball 估计背景并扣除，更接近 ImageJ。
    需要: pip install scikit-image
    """
    try:
        from skimage.restoration import rolling_ball
    except Exception as e:
        raise RuntimeError("需要安装 scikit-image 才能使用 --backend skimage，请先运行: pip install -U scikit-image") from e

    img_f = img_u8.astype(np.float32)
    background = rolling_ball(img_f, radius=radius)
    corrected = img_f - background
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected


def _ensure_imagej():
    global _IJ_CTX, _IJ_CLASS
    if _IJ_CTX is not None and _IJ_CLASS is not None:
        return _IJ_CTX, _IJ_CLASS
    try:
        import imagej  # type: ignore
        from scyjava import jimport  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "需要安装 PyImageJ 才能使用 --backend imagej\n"
            "请先运行: pip install -U pyimagej scyjava\n"
            "首次运行会自动下载 Fiji 依赖，需联网"
        ) from e
    _IJ_CTX = imagej.init("sc.fiji:fiji", headless=True)
    _IJ_CLASS = jimport('ij.IJ')
    return _IJ_CTX, _IJ_CLASS


def subtract_background_imagej(img_u8: np.ndarray, radius: int, sliding: bool = True) -> np.ndarray:
    """调用 Fiji 的 IJ1 命令 Subtract Background... 实现精确一致的结果。
    实现方式：将 numpy 临时写入 TIFF → IJ.openImage → IJ.run("Subtract Background...", ...) → 保存 → 读回 numpy。
    这样可避免复杂的 Java-对象转换，保证兼容性。
    """
    ij_ctx, IJ = _ensure_imagej()

    # 写临时输入与输出
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as in_f:
        in_path = in_f.name
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as out_f:
        out_path = out_f.name

    try:
        # 写入 8-bit TIFF
        ok = cv2.imwrite(in_path, img_u8)
        if not ok:
            raise RuntimeError("写入临时文件失败")

        imp = IJ.openImage(in_path)
        if imp is None:
            raise RuntimeError("ImageJ 无法打开临时图像")

        arg = f"rolling={int(radius)}"
        if sliding:
            arg += " sliding"
        IJ.run(imp, "Subtract Background...", arg)
        IJ.saveAs(imp, "Tiff", out_path)
        imp.close()

        # 读回
        result = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        if result is None:
            raise RuntimeError("读取处理结果失败")
        return result
    finally:
        # 清理临时文件
        try:
            os.remove(in_path)
        except Exception:
            pass
        try:
            os.remove(out_path)
        except Exception:
            pass


def process_one(jpg_path: Path, output_dir: Path, rolling: int, backend: str, sliding: bool) -> tuple[Path, bool, str]:
    try:
        img = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)
        if img is None:
            return jpg_path, False, "读取失败"

        # 蓝通道（OpenCV 为 BGR 顺序）
        blue = img[:, :, 0]

        # 反相
        inv = cv2.subtract(255, blue)

        # 背景扣除
        if backend == "imagej":
            corrected = subtract_background_imagej(inv, rolling, sliding=sliding)
        elif backend == "skimage":
            corrected = subtract_background_skimage(inv, rolling)
        else:
            corrected = subtract_background_morphology(inv, rolling)

        # 保存为单通道 8-bit TIFF
        out_path = output_dir / f"processed_{jpg_path.stem}.tif"
        ok = cv2.imwrite(str(out_path), corrected)
        return out_path, bool(ok), ""
    except Exception as e:
        return jpg_path, False, str(e)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    rolling = int(args.rolling)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在或不可用: {input_dir}")

    ensure_dir(output_dir)
    jpgs = list_jpgs(input_dir)

    # 参数校验：分片
    if args.num_shards < 1:
        raise SystemExit("--num-shards 必须 >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard-index 必须在 [0, --num-shards) 范围内")

    # 分片过滤：仅保留当前分片的文件
    if args.num_shards > 1:
        jpgs = [p for i, p in enumerate(jpgs) if i % args.num_shards == args.shard_index]

    if not jpgs:
        print("未在输入目录发现 .jpg 文件（仅扫描顶层目录）")
        return

    if args.workers == "auto":
        workers = max(1, (mp.cpu_count() or 2) - 1)
    else:
        try:
            workers = max(1, int(args.workers))
        except ValueError:
            workers = 1

    # ImageJ 后端不支持并发线程：JVM 与 IJ1 命令存在全局状态，线程不安全
    if args.backend == "imagej" and workers != 1:
        print("警告: --backend imagej 不支持并发，已自动将 --workers 调整为 1 以避免崩溃/卡死/结果异常。")
        workers = 1

    print(f"待处理: {len(jpgs)} 张 .jpg | 线程: {workers} | rolling半径: {rolling} px | 后端: {args.backend} | 分片: {args.shard_index+1}/{args.num_shards}")

    successes = 0
    failures = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(process_one, p, output_dir, rolling, args.backend, (not args.no_sliding))
            for p in jpgs
        ]
        for idx, fut in enumerate(as_completed(futures), 1):
            out_path, ok, msg = fut.result()
            if ok:
                successes += 1
                if idx % 20 == 0 or idx == len(jpgs):
                    print(f"[{idx}/{len(jpgs)}] 已保存: {out_path.name}")
            else:
                failures += 1
                print(f"处理失败: {out_path} | 原因: {msg}")

    print(f"完成。成功: {successes}, 失败: {failures}, 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
