import cv2, numpy as np, os

gt_path   = r"D:\project\2-spring\DATASET6\data\DAY0\images\R-D0-1.png"          # 手工标注（二值）
pred_path = r"D:\project\2-spring\DATASET6\data\DAY0\images\R-D0-1_mask.png" # 模型分割（二值）

def read_bin_mask(p):
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(p)
    # 二值化：非零视为前景。若你的前景是黑色，请反相：m = 255 - m
    return (m > 127).astype(np.uint8)

def annotate_components(binary_mask: np.ndarray,
                        canvas_bgr: np.ndarray,
                        text_bgr_color: tuple,
                        prefix: str = 'roi',
                        min_area: int = 30,
                        font_scale: float = 0.5):
    """
    在 canvas_bgr 上对 binary_mask 的每个连通域标注 roi1、roi2…
    - binary_mask: 0/1 二值（前景=1）
    - canvas_bgr: 与 mask 尺寸一致的三通道 BGR 图像
    - text_bgr_color: 文本颜色（B,G,R）
    - prefix: 文本前缀，默认 'roi'
    - min_area: 过滤太小的噪点区域
    - font_scale: 文本尺寸
    """
    foreground = (binary_mask.astype(np.uint8) > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    roi_index = 1
    for label_id in range(1, num_labels):  # 0 是背景
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = centroids[label_id]
        x, y = int(cx), int(cy)
        text = f"{prefix}{roi_index}"
        roi_index += 1
        # 先描黑边，再用目标颜色细描，增强可读性
        cv2.putText(canvas_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_bgr_color, 1, cv2.LINE_AA)

gt = read_bin_mask(gt_path)
pred = read_bin_mask(pred_path)

# 尺寸不一致则按最近邻对齐
if pred.shape != gt.shape:
    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

tp = int(np.sum((gt==1) & (pred==1)))
fp = int(np.sum((gt==0) & (pred==1)))
fn = int(np.sum((gt==1) & (pred==0)))
tn = int(np.sum((gt==0) & (pred==0)))
eps = 1e-10

precision = tp / (tp + fp + eps)
recall    = tp / (tp + fn + eps)
dice      = (2*tp) / (2*tp + fp + fn + eps)
iou       = tp / (tp + fp + fn + eps)
accuracy  = (tp + tn) / (tp + tn + fp + fn + eps)

print(f"IoU={iou:.4f}, Dice={dice:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Accuracy={accuracy:.4f}")

# ---------- 可视化：GT vs Pred 重叠对比 ----------
h, w = gt.shape
overlay = np.zeros((h, w, 3), dtype=np.uint8)

# BGR 颜色：
# TP (gt=1 & pred=1) -> 绿色
overlay[(gt==1) & (pred==1)] = (0, 255, 0)
# FP (gt=0 & pred=1) -> 红色（模型多分）
overlay[(gt==0) & (pred==1)] = (0, 0, 255)
# FN (gt=1 & pred==0) -> 蓝色（模型漏分）
overlay[(gt==1) & (pred==0)] = (255, 0, 0)

# 也生成 GT 与 Pred 的彩色可视化，便于左右对比
vis_gt = np.zeros((h, w, 3), dtype=np.uint8)
vis_gt[gt==1] = (0, 255, 0)   # 绿色
vis_pred = np.zeros((h, w, 3), dtype=np.uint8)
vis_pred[pred==1] = (0, 0, 255)  # 红色

# 在 GT 与 Pred 面板上分别标注连通域编号（roi1、roi2…）
annotate_components(gt, vis_gt, (0, 180, 0), prefix='roi', min_area=30, font_scale=0.5)
annotate_components(pred, vis_pred, (0, 0, 180), prefix='roi', min_area=30, font_scale=0.5)

# 组合成三联图：GT | Overlay | Pred
pad = np.ones((h, 10, 3), dtype=np.uint8) * 255
panel = np.concatenate([vis_gt, pad, overlay, pad, vis_pred], axis=1)

# 在图上写文字说明
cv2.putText(panel, 'GT (green)', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 180, 60), 2)
cv2.putText(panel, 'Overlay: TP=Green, FP=Red, FN=Blue', (w+20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
cv2.putText(panel, 'Pred (red)', (2*w+30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,180), 2)

out_dir = os.path.dirname(pred_path)
out_path = os.path.join(out_dir, 'compare_overlay.png')
cv2.imwrite(out_path, panel)
print(f'Saved overlay to: {out_path}')