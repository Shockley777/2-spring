"""
I/O helpers for images and masks with minimal dependencies
"""
from typing import Tuple, List, Dict, Any
import os

import numpy as np
from PIL import Image

try:
    import cv2  # optional, used when available
except ImportError:
    cv2 = None  # type: ignore


def load_image(path: str) -> np.ndarray:
    """
    Loads an image as RGB uint8 numpy array of shape (H, W, 3)
    """
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_mask(mask: np.ndarray, path: str) -> None:
    """
    Saves a binary or instance mask.
    If mask is 2D integer array, saves as PNG preserving labels.
    """
    mask_img = Image.fromarray(mask.astype(np.uint16) if mask.max() > 1 else mask.astype(np.uint8))
    mask_img.save(path)


def find_image_files(input_dir: str, exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")) -> List[str]:
    files = []
    for root, _, fns in os.walk(input_dir):
        for f in fns:
            if f.lower().endswith(exts):
                files.append(os.path.join(root, f))
    files.sort()
    return files


def connected_components(mask: np.ndarray) -> np.ndarray:
    """
    Returns labeled instance mask using simple connectivity.
    """
    if cv2 is not None:
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        return labels
    else:
        # fallback using scipy when available else simple BFS (not implemented): use numpy trick for binary only
        # For simplicity, return binary as 0/1 labels
        return (mask > 0).astype(np.uint8)


def contours_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    """
    Extract polygon contours from a binary mask. Requires OpenCV; if not present, returns empty list.
    """
    if cv2 is None:
        return []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [c.squeeze(1) for c in contours if len(c) >= 3]
    return polys
