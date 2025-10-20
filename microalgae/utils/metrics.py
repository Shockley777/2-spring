"""
Segmentation and contrastive learning metrics
- Dice, IoU for segmentation
- InfoNCE loss for contrastive alignment
Both implement CPU/GPU agnostic torch versions with no external deps
"""
from typing import Tuple
import torch
import torch.nn.functional as F


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes Sørensen–Dice coefficient for binary masks.
    Args:
        pred: (N, 1, H, W) or (N, H, W) predicted probabilities in [0,1]
        target: (N, 1, H, W) or (N, H, W) binary ground-truth {0,1}
    Returns:
        Dice score averaged over batch
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    pred_bin = (pred > 0.5).float()
    target = target.float()
    inter = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean()


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Intersection over Union for binary masks.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    pred_bin = (pred > 0.5).float()
    target = target.float()
    inter = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft dice loss using probabilities (sigmoid on logits)
    """
    probs = torch.sigmoid(logits)
    if probs.dim() == 3:
        probs = probs.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    target = target.float()
    inter = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    dice = soft_dice_loss(logits, target)
    return bce_weight * bce + (1 - bce_weight) * dice


def info_nce_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    InfoNCE loss between two sets of embeddings (batch-aligned pairs)
    Args:
        z_i: (N, D) modality A
        z_j: (N, D) modality B
    Returns:
        scalar loss
    """
    z_i = F.normalize(z_i, dim=-1)
    z_j = F.normalize(z_j, dim=-1)
    logits = z_i @ z_j.t() / temperature  # (N, N)
    labels = torch.arange(z_i.size(0), device=z_i.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_j) / 2
