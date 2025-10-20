"""
微藻高精度分割 (High-precision segmentation) 基于 SAM + PEFT (Adapters)

- 尝试优先使用 Segment Anything 的图像编码器作为冻结主干
- 在其输出特征上追加轻量上采样头 (PEFT Head)，仅训练少量参数
- 若不可用（无 segment-anything 依赖），退化为轻量 U-Net

- 训练包含多种鲁棒性评估：
  - 领域偏移（颜色/对比度变化）
  - 低信噪比（高斯噪声）
  - 形态变化（模糊/形态学扰动）

输入标注可以来自半自动标注产出的 PNG 实例掩码（自动合并为二值目标）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import os

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from ..utils.metrics import dice_coefficient, iou_score, bce_dice_loss
from ..utils.io import load_image


# -------------------------- Models --------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, k, padding=p), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class UNetLite(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.head(d1)


class Adapter(nn.Module):
    """Simple bottleneck adapter for PEFT."""
    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Conv2d(dim, bottleneck, 1)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(bottleneck, dim, 1)
    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class SAMBackboneWrapper(nn.Module):
    """
    Wraps SAM image encoder when available. If not, raises ImportError.
    Only exposes image_encoder forward to get feature map.
    """
    def __init__(self, checkpoint: str, model_type: str = "vit_b"):
        super().__init__()
        from segment_anything import sam_model_registry  # type: ignore
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.image_encoder = sam.image_encoder
        for p in self.image_encoder.parameters():
            p.requires_grad = False  # freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SAM expects normalized images in [0,1] with specific resizing; for simplicity, rely on caller transforms
        return self.image_encoder(x)


class SAMPEFTHead(nn.Module):
    """
    Lightweight upsampling head with adapters. Input: encoder feature map (B,C,h,w)
    """
    def __init__(self, in_ch: int, mid: int = 256, adapters: int = 2):
        super().__init__()
        layers: List[nn.Module] = [nn.Conv2d(in_ch, mid, 1), nn.ReLU(inplace=True)]
        for _ in range(adapters):
            layers.append(Adapter(mid, bottleneck=64))
        self.proj = nn.Sequential(*layers)
        self.up1 = nn.ConvTranspose2d(mid, mid // 2, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(mid // 2, mid // 4, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(mid // 4, mid // 8, 2, stride=2)
        self.head = nn.Conv2d(max(1, mid // 8), 1, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        x = self.proj(f)
        x = self.up1(x)
        x = F.relu(x, inplace=True)
        x = self.up2(x)
        x = F.relu(x, inplace=True)
        x = self.up3(x)
        x = F.relu(x, inplace=True)
        return self.head(x)


class SAMPEFTSegmentation(nn.Module):
    def __init__(self, sam_checkpoint: Optional[str] = None, model_type: str = "vit_b"):
        super().__init__()
        self.using_sam = False
        self.backbone: Optional[nn.Module] = None
        self.normalizer = T.Normalize(mean=[123.675/255, 116.28/255, 103.53/255], std=[58.395/255, 57.12/255, 57.375/255])
        if sam_checkpoint is not None:
            try:
                self.backbone = SAMBackboneWrapper(checkpoint=sam_checkpoint, model_type=model_type)
                # Probe a dummy to infer channels: SAM ViT-B returns (B,256,64,64) for 1024 input
                in_ch = 256
                self.head = SAMPEFTHead(in_ch=in_ch)
                self.using_sam = True
            except Exception:
                self.backbone = None
        if self.backbone is None:
            # fallback to simple UNet
            self.head = UNetLite(in_ch=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.using_sam and self.backbone is not None:
            x = x.float() / 255.0
            x = self.normalizer(x)
            feats = self.backbone(x)
            logits = self.head(feats)
        else:
            logits = self.head(x.float() / 255.0)
        return logits


# -------------------------- Dataset --------------------------
class SegmentationDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, size: int = 512):
        super().__init__()
        self.image_paths = []
        for fn in os.listdir(image_dir):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                self.image_paths.append(os.path.join(image_dir, fn))
        self.image_paths.sort()
        self.mask_dir = mask_dir
        self.tf_img = T.Compose([T.ToTensor(), T.Resize((size, size), antialias=True)])
        self.tf_mask = T.Compose([T.Resize((size, size), antialias=True)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        ip = self.image_paths[idx]
        mp = os.path.join(self.mask_dir, os.path.splitext(os.path.basename(ip))[0] + "_mask.png")
        img = load_image(ip)
        mask = np.array(Image.open(mp))
        # convert instance mask to binary
        mask = (mask > 0).astype(np.uint8) * 255
        img_t = (self.tf_img(Image.fromarray(img)) * 255.0).byte()
        mask_t = torch.from_numpy(np.array(self.tf_mask(Image.fromarray(mask))))[None, ...] / 255.0
        return img_t, mask_t.float()


# -------------------------- Training & Eval --------------------------
@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 4
    epochs: int = 20
    num_workers: int = 2
    bce_weight: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader], cfg: TrainConfig) -> Dict[str, float]:
    model.to(cfg.device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=1e-4)
    best_val = 0.0
    history = {"val_dice": 0.0, "val_iou": 0.0}
    for epoch in range(cfg.epochs):
        model.train()
        for img, mask in train_loader:
            img = img.to(cfg.device)
            mask = mask.to(cfg.device)
            logits = model(img)
            # resize logits to mask size if needed
            if logits.shape[-2:] != mask.shape[-2:]:
                logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            loss = bce_dice_loss(logits, mask, bce_weight=cfg.bce_weight)
            opt.zero_grad(); loss.backward(); opt.step()
        # Eval
        if val_loader is not None:
            model.eval()
            dices, ious = [], []
            with torch.no_grad():
                for img, mask in val_loader:
                    img = img.to(cfg.device)
                    mask = mask.to(cfg.device)
                    logits = model(img)
                    if logits.shape[-2:] != mask.shape[-2:]:
                        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
                    probs = torch.sigmoid(logits)
                    dices.append(dice_coefficient(probs, mask).item())
                    ious.append(iou_score(probs, mask).item())
            mean_dice = float(np.mean(dices)) if dices else 0.0
            mean_iou = float(np.mean(ious)) if ious else 0.0
            history = {"val_dice": mean_dice, "val_iou": mean_iou}
            if mean_dice > best_val:
                best_val = mean_dice
        print(f"Epoch {epoch+1}/{cfg.epochs} - val_dice={history['val_dice']:.4f} val_iou={history['val_iou']:.4f}")
    return history


def evaluate_robustness(model: nn.Module, data_loader: DataLoader, device: str) -> Dict[str, float]:
    """
    对领域偏移、低信噪比、形态变化进行鲁棒性评估
    """
    model.to(device)
    model.eval()
    def _eval_with_transform(img: torch.Tensor, mask: torch.Tensor, tf: T.Compose) -> Tuple[float, float]:
        dices, ious = [], []
        with torch.no_grad():
            for x, y in zip(img, mask):
                xt = tf(x)
                xt = xt[None].to(device)
                yt = y[None].to(device)
                logits = model(xt)
                if logits.shape[-2:] != yt.shape[-2:]:
                    logits = F.interpolate(logits, size=yt.shape[-2:], mode="bilinear", align_corners=False)
                probs = torch.sigmoid(logits)
                dices.append(dice_coefficient(probs, yt).item())
                ious.append(iou_score(probs, yt).item())
        return float(np.mean(dices)), float(np.mean(ious))

    # Collect a batch
    imgs, masks = next(iter(data_loader))
    # Domain shift: color jitter
    jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
    d_dice, d_iou = _eval_with_transform(imgs, masks, T.Compose([jitter]))
    # Low SNR: gaussian noise
    def add_noise(x: torch.Tensor):
        noise = torch.randn_like(x) * 15.0  # assuming [0,255]
        return (x + noise).clamp(0, 255)
    n_dice, n_iou = _eval_with_transform(imgs, masks, T.Compose([T.Lambda(add_noise)]))
    # Morphological: blur
    blur = T.GaussianBlur(5, sigma=(1.0, 2.0))
    m_dice, m_iou = _eval_with_transform(imgs, masks, T.Compose([blur]))

    return {
        "domain_shift_dice": d_dice, "domain_shift_iou": d_iou,
        "low_snr_dice": n_dice, "low_snr_iou": n_iou,
        "morphology_dice": m_dice, "morphology_iou": m_iou,
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="Train SAM+PEFT segmentation or fallback UNet")
    p.add_argument("--images", required=True, help="Directory with raw images")
    p.add_argument("--masks", required=True, help="Directory with binary/instance masks named <image>_mask.png")
    p.add_argument("--sam-checkpoint", default=None, help="Path to SAM checkpoint")
    p.add_argument("--sam-type", default="vit_b")
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args()

    ds = SegmentationDataset(args.images, args.masks, size=512)
    n = len(ds)
    n_tr = int(0.8 * n)
    tr, va = torch.utils.data.random_split(ds, [n_tr, n - n_tr])
    tl = DataLoader(tr, batch_size=4, shuffle=True, num_workers=2)
    vl = DataLoader(va, batch_size=4, shuffle=False, num_workers=2)

    model = SAMPEFTSegmentation(sam_checkpoint=args.sam_checkpoint, model_type=args.sam_type)
    hist = train(model, tl, vl, TrainConfig(epochs=args.epochs))
    print("Validation:", hist)
    robust = evaluate_robustness(model, vl, device=TrainConfig().device)
    print("Robustness:", robust)


if __name__ == "__main__":
    main()
