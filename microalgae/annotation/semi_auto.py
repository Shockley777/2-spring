"""
半自动标注管道 (Semi-Automatic Annotation Pipeline)

目标：
- 通过预标注模型（如SAM）+ 弱监督聚合，降低显微图像的人工标注成本
- 提供系统化的数据增强与预处理，以提升微藻图像的可分割性
- 产出：实例掩码（PNG）+ 轮廓与bbox元数据（JSON），并保留人工修正接口

依赖：
- 核心依赖：numpy, torch, torchvision, PIL
- 可选依赖：opencv-python, segment-anything, timm, transformers
  当可选依赖不存在时，自动退化为传统图像处理预分割（Otsu + 形态学）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
import json

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

import torch
from torchvision import transforms as T

from ..utils.io import load_image, save_mask, find_image_files, connected_components, contours_from_mask

try:  # optional
    import cv2
except Exception:
    cv2 = None  # type: ignore


@dataclass
class PreprocessConfig:
    """预处理配置 Preprocessing parameters"""
    resize: Optional[int] = 1024
    clahe: bool = True
    denoise_ksize: int = 3


@dataclass
class AugmentConfig:
    """数据增强配置 Augmentations"""
    hflip: bool = True
    vflip: bool = False
    rotate_deg: int = 15
    color_jitter: bool = True


class ImagePreprocessor:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # Resize keeping aspect
        if self.cfg.resize is not None:
            pil = Image.fromarray(img)
            pil = ImageOps.contain(pil, (self.cfg.resize, self.cfg.resize))
            img = np.array(pil)
        # CLAHE for contrast enhancement
        if self.cfg.clahe and cv2 is not None:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        elif self.cfg.clahe:
            pil = Image.fromarray(img)
            enhancer = ImageEnhance.Contrast(pil)
            img = np.array(enhancer.enhance(1.3))
        # Light denoise
        if cv2 is not None and self.cfg.denoise_ksize > 0:
            k = self.cfg.denoise_ksize
            img = cv2.GaussianBlur(img, (k if k % 2 == 1 else k + 1, k if k % 2 == 1 else k + 1), 0)
        return img


class DataAugmenter:
    def __init__(self, cfg: AugmentConfig):
        ops = []
        if cfg.hflip:
            ops.append(T.RandomHorizontalFlip(p=0.5))
        if cfg.vflip:
            ops.append(T.RandomVerticalFlip(p=0.5))
        if cfg.rotate_deg and cfg.rotate_deg > 0:
            ops.append(T.RandomRotation(cfg.rotate_deg))
        if cfg.color_jitter:
            ops.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02))
        self.tf = T.Compose(ops)

    def __call__(self, img: np.ndarray, n: int = 2) -> List[np.ndarray]:
        imgs = []
        pil = Image.fromarray(img)
        for _ in range(n):
            aug = self.tf(pil)
            imgs.append(np.array(aug))
        return imgs


class PreAnnotator:
    """
    预标注器：优先使用 SAM 的自动掩码生成；否则退回 Otsu 阈值 + 形态学
    """
    def __init__(self, sam_checkpoint: Optional[str] = None, model_type: str = "vit_b"):
        self.use_sam = False
        self.mask_generator = None
        if sam_checkpoint is not None:
            try:
                from segment_anything import sam_model_registry, SamAutomaticMaskGenerator  # type: ignore
                sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                sam.eval()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                sam.to(device)
                self.mask_generator = SamAutomaticMaskGenerator(sam)
                self.use_sam = True
            except Exception:
                self.use_sam = False

    def _fallback_binary(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if cv2 is not None else np.array(Image.fromarray(img).convert("L"))
        if cv2 is not None:
            thr, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = (mask > 0).astype(np.uint8)
            # Morph open-close to handle adhesion
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        else:
            # Simple threshold
            thr = gray.mean()
            mask = (gray > thr).astype(np.uint8)
        return mask

    def __call__(self, img: np.ndarray) -> List[np.ndarray]:
        if self.use_sam and self.mask_generator is not None:
            masks = self.mask_generator.generate(img)
            masks_sorted = sorted(masks, key=lambda m: m.get("area", 0), reverse=True)
            binary_masks = [(m["segmentation" ]).astype(np.uint8) for m in masks_sorted]
            return binary_masks
        else:
            return [self._fallback_binary(img)]


class WeakSupervisor:
    """简单弱监督聚合：通过多视图一致性构建高置信掩码"""
    def __init__(self, min_consensus: int = 2):
        self.min_consensus = min_consensus

    def aggregate(self, candidates: List[np.ndarray]) -> np.ndarray:
        if not candidates:
            raise ValueError("No candidate masks provided")
        h, w = candidates[0].shape
        stack = np.stack([c.astype(np.uint8) for c in candidates], axis=0)
        votes = stack.sum(axis=0)
        agg = (votes >= self.min_consensus).astype(np.uint8)
        return agg


class SemiAutoAnnotationPipeline:
    def __init__(self, preprocess: PreprocessConfig, augment: AugmentConfig, sam_checkpoint: Optional[str] = None, model_type: str = "vit_b"):
        self.preproc = ImagePreprocessor(preprocess)
        self.augmenter = DataAugmenter(augment)
        self.pre_annotator = PreAnnotator(sam_checkpoint=sam_checkpoint, model_type=model_type)
        self.weak = WeakSupervisor(min_consensus=2)

    def annotate_image(self, img: np.ndarray) -> Dict[str, Any]:
        # 预处理
        img_p = self.preproc(img)
        # 多视图增强
        views = [img_p] + self.augmenter(img_p, n=2)
        # 候选掩码
        candidates: List[np.ndarray] = []
        for v in views:
            masks = self.pre_annotator(v)
            # 仅保留大目标候选，减少噪声
            if masks:
                topk = masks[:10]
                # 合并多个实例为一张二值图（保留细胞整体）
                bin_union = np.clip(np.sum(np.stack(topk, axis=0), axis=0), 0, 1).astype(np.uint8)
                candidates.append(bin_union)
        # 弱监督聚合
        agg = self.weak.aggregate(candidates)
        # 连通域 -> 实例标签
        inst = connected_components(agg)
        # 提取边界与bbox
        contours = contours_from_mask(agg)
        bboxes = []
        if cv2 is not None:
            for c in contours:
                x, y, w, h = cv2.boundingRect(c.reshape(-1, 1, 2))
                bboxes.append([int(x), int(y), int(w), int(h)])
        meta = {
            "num_instances": int(inst.max()),
            "bboxes": bboxes,
            "contours": [c.reshape(-1, 2).tolist() for c in contours],
        }
        return {"mask": inst.astype(np.uint16), "meta": meta}

    def run(self, input_dir: str, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
        images = find_image_files(input_dir)
        records: List[Dict[str, Any]] = []
        for path in images:
            img = load_image(path)
            res = self.annotate_image(img)
            base = os.path.splitext(os.path.basename(path))[0]
            mask_path = os.path.join(output_dir, "masks", f"{base}_mask.png")
            save_mask(res["mask"], mask_path)
            record = {
                "image": os.path.relpath(path, input_dir),
                "mask": os.path.relpath(mask_path, output_dir),
                "meta": res["meta"],
            }
            records.append(record)
        with open(os.path.join(output_dir, "annotations.json"), "w", encoding="utf-8") as f:
            json.dump({"annotations": records}, f, ensure_ascii=False, indent=2)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Microalgae semi-automatic annotation")
    p.add_argument("--input", required=True, help="Input directory with images")
    p.add_argument("--output", required=True, help="Output directory for masks and JSON")
    p.add_argument("--sam-checkpoint", default=None, help="Path to SAM checkpoint .pth")
    p.add_argument("--sam-type", default="vit_b", help="SAM backbone type: vit_b/vit_l/vit_h")
    args = p.parse_args()

    pipeline = SemiAutoAnnotationPipeline(
        preprocess=PreprocessConfig(),
        augment=AugmentConfig(),
        sam_checkpoint=args.sam_checkpoint,
        model_type=args.sam_type,
    )
    pipeline.run(args.input, args.output)


if __name__ == "__main__":
    main()
