"""
跨模态特征融合：GNN + 对比学习 (Multimodal Feature Fusion)

- 图构建：基于kNN从任一模态或联合特征构图
- 模态编码：图像编码器(ResNet18) + 组学编码器(MLP)
- 对比学习：InfoNCE 对齐两种表征（弱配准场景用批内对齐）
- GNN 融合：在样本图上做消息传递，得到可解释的联合表示

依赖：torch, torchvision, pandas, numpy
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import os

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T

from ..utils.metrics import info_nce_loss


class MultimodalDataset(Dataset):
    """
    读取包含图像路径和组学向量的CSV：
    - CSV列: image, f0, f1, ..., f{D-1}
    """
    def __init__(self, csv_path: str, image_root: Optional[str] = None, size: int = 224):
        super().__init__()
        df = pd.read_csv(csv_path)
        self.image_paths = df["image"].tolist()
        if image_root is not None:
            self.image_paths = [os.path.join(image_root, p) for p in self.image_paths]
        self.omics = df.drop(columns=["image"]).values.astype(np.float32)
        self.tf = T.Compose([
            T.Resize((size, size), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        x = self.tf(img)
        o = torch.from_numpy(self.omics[idx])
        return x, o


# ---------- Encoders ----------
class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        # Avoid internet downloads; use random init
        base = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*(list(base.children())[:-1]))  # (B,512,1,1)
        self.fc = nn.Linear(512, out_dim)
    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.fc(f)


class OmicsEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)


# ---------- Simple GNN (Adjacency-based message passing) ----------
class GraphSageLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (N,D), adj: (N,N) normalized
        neigh = adj @ x
        h = torch.cat([x, neigh], dim=-1)
        return F.relu(self.lin(h), inplace=True)


class SimpleGNN(nn.Module):
    def __init__(self, dim: int, layers: int = 2):
        super().__init__()
        mods = []
        d = dim
        for _ in range(layers):
            mods.append(GraphSageLayer(d, d))
        self.layers = nn.ModuleList(mods)
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = l(x, adj)
        return x


# ---------- Trainer ----------
@dataclass
class FusionConfig:
    batch_size: int = 16
    lr: float = 1e-3
    epochs: int = 20
    k: int = 5  # kNN for graph
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    # x: (N,D)
    with torch.no_grad():
        x = F.normalize(x, dim=-1)
        sim = x @ x.t()  # cosine
        vals, idx = torch.topk(sim, k=k + 1, dim=1)  # include self
        N = x.size(0)
        adj = torch.zeros(N, N, device=x.device)
        for i in range(N):
            for j in idx[i].tolist():
                if i != j:
                    adj[i, j] = 1.0
        # symmetrize and normalize
        adj = torch.maximum(adj, adj.t())
        deg = adj.sum(dim=1, keepdim=True) + 1e-6
        adj = adj / deg
        return adj


def train(csv_path: str, image_root: Optional[str], cfg: FusionConfig) -> Dict[str, float]:
    ds = MultimodalDataset(csv_path, image_root=image_root)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    # infer omics dim
    omics_dim = ds.omics.shape[1]
    img_enc = ImageEncoder(out_dim=256).to(cfg.device)
    omic_enc = OmicsEncoder(in_dim=omics_dim, out_dim=256).to(cfg.device)
    gnn = SimpleGNN(dim=256, layers=2).to(cfg.device)
    opt = torch.optim.AdamW(list(img_enc.parameters()) + list(omic_enc.parameters()) + list(gnn.parameters()), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        img_enc.train(); omic_enc.train(); gnn.train()
        tot = 0.0
        for img, om in dl:
            img = img.to(cfg.device)
            om = om.to(cfg.device)
            zi = img_enc(img)
            zj = omic_enc(om)
            # graph on joint
            adj = knn_graph(torch.cat([zi, zj], dim=0), k=cfg.k)
            # message passing
            z = torch.cat([zi, zj], dim=0)
            z_g = gnn(z, adj)
            zi_g, zj_g = z_g[: zi.size(0)], z_g[zi.size(0) :]
            # contrastive loss
            loss = info_nce_loss(zi_g, zj_g, temperature=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"Epoch {epoch+1}/{cfg.epochs} - CL loss: {tot / max(1, len(dl)):.4f}")

    # Return a simple alignment score on last batch
    img_enc.eval(); omic_enc.eval(); gnn.eval()
    with torch.no_grad():
        img, om = next(iter(dl))
        img = img.to(cfg.device); om = om.to(cfg.device)
        zi = img_enc(img); zj = omic_enc(om)
        adj = knn_graph(torch.cat([zi, zj], dim=0), k=cfg.k)
        z_g = gnn(torch.cat([zi, zj], dim=0), adj)
        zi_g, zj_g = z_g[: zi.size(0)], z_g[zi.size(0) :]
        # cosine similarities for positive pairs
        sims = F.cosine_similarity(F.normalize(zi_g, dim=-1), F.normalize(zj_g, dim=-1)).mean().item()
    return {"alignment_cosine": sims}


def main():
    import argparse
    p = argparse.ArgumentParser(description="Train multimodal fusion with GNN + contrastive learning")
    p.add_argument("--pairs-csv", required=True, help="CSV with columns: image,f0,f1,...")
    p.add_argument("--image-root", default=None, help="Optional root to prepend to image paths")
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args()
    stats = train(args.pairs_csv, args.image_root, FusionConfig(epochs=args.epochs))
    print("Alignment:", stats)


if __name__ == "__main__":
    main()
