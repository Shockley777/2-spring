"""
Microalgae Research Toolkit

This package provides:
1) Semi-automatic annotation pipeline combining pre-annotation models (e.g., SAM) with weak supervision and expert correction hooks
2) High-precision segmentation framework based on Segment Anything with parameter-efficient fine-tuning (PEFT Adapters)
3) Multimodal feature fusion using simple GNN and contrastive learning for cross-scale interpretability

Note: Heavy optional dependencies (segment-anything, peft, timm, transformers, opencv) are imported lazily when used.
"""

__all__ = [
    "annotation",
    "segmentation",
    "fusion",
    "utils",
]
