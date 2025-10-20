# Microalgae Research Toolkit (Extended)

This repository now includes three major components for microalgae image analysis and multimodal learning:

1) Semi-Automatic Annotation System
- Goal: reduce expert annotation cost by combining pre-annotation models with weak supervision and expert correction.
- Pipeline: preprocessing + data augmentation + pre-annotation (SAM when available; fallback to classical image processing) + weak supervision aggregation.
- Output: instance masks (PNG) plus metadata (JSON with contours and bounding boxes) that can be corrected by experts.

2) High-Precision Segmentation Model (SAM + PEFT)
- Segment Anything image encoder is used as a frozen backbone when available.
- A lightweight PEFT adapter head is trained on top of SAM features to produce full-resolution masks.
- If SAM is not available, a UNet-lite fallback is used.
- Includes robustness evaluation to domain shift, low SNR, and morphological variations.

3) Multimodal Feature Fusion (GNN + Contrastive Learning)
- Aligns microscopy image features with heterogeneous data (omics, spectroscopy) using contrastive learning and a simple GNN for message passing on a kNN graph.
- Produces an interpretable joint representation that can be used for cross-scale inference.


Getting Started

Environment
- Python >= 3.8
- Core deps already in requirements.txt: numpy, pandas, torch, torchvision, etc.
- Optional deps (only needed for enhanced functionality):
  - segment-anything (Facebook Research)
  - opencv-python
  - timm / transformers (for advanced backbones, if desired)

The code imports optional packages lazily. If they are missing, the pipeline falls back to classical vision components.


1) Semi-Automatic Annotation

Run
- Prepare an input directory with raw microscope images.
- Execute:
  python -m microalgae.annotation.semi_auto --input /path/to/images --output /path/to/annotations \
      --sam-checkpoint /path/to/sam_vit_b.pth --sam-type vit_b

Notes
- --sam-checkpoint is optional. When provided, the pipeline uses SAM's automatic mask generator to generate proposals.
- The output directory contains:
  - masks/<image>_mask.png: instance-labeled mask (uint16)
  - annotations.json: metadata with bounding boxes and polygon contours


2) Segmentation: SAM + PEFT Adapters

Data
- Use the outputs from the semi-automatic annotation as pseudo-labels, optionally refined by experts.
- Put raw images under: /path/to/images
- Put masks under: /path/to/annotations/masks (file naming: <image>_mask.png)

Train
  python -m microalgae.segmentation.sam_peft --images /path/to/images --masks /path/to/annotations/masks \
      --sam-checkpoint /path/to/sam_vit_b.pth --sam-type vit_b --epochs 20

- If SAM checkpoint is not provided, the code trains a UNet-lite fallback.
- After training, the script reports validation Dice/IoU and robustness scores against synthetic domain shifts.


3) Multimodal Fusion: GNN + Contrastive

Data CSV
- Prepare a CSV with columns: image,f0,f1,...,fD-1
  - image: relative or absolute image path
  - f*: omics/spectroscopy features per sample

Train
  python -m microalgae.fusion.multimodal_gnn_contrastive --pairs-csv /path/to/pairs.csv --image-root /optional/image/root --epochs 20

- The script prints a batch-level alignment cosine similarity after training, reflecting cross-modal alignment quality.


Design Notes
- Semi-Automatic Annotation mixes pre-annotation (SAM or Otsu+morphology) with weak supervision via multi-view consensus to handle low SNR and adhesion.
- Segmentation model adapts SAM with parameter-efficient adapters to the microalgae domain, enabling high-precision instance segmentation with limited labels.
- Multimodal fusion uses a kNN graph and simple GNN layers with contrastive alignment, supporting weak or missing registration by relying on in-batch pairing.


Repository Structure Additions
- microalgae/
  - annotation/semi_auto.py
  - segmentation/sam_peft.py
  - fusion/multimodal_gnn_contrastive.py
  - utils/
    - metrics.py
    - io.py

Existing project files for tabular experiments are kept unchanged.
