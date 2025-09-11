# DATASET5 使用说明（标准流程）

本说明以 DATASET5 为标准，给出从数据组织、ImageJ 等价预处理、分割、特征提取、合并与可视化的完整流程。所有命令均在项目根目录执行：`D:\project\2-spring`。

## 目录结构
建议按 Day 划分（最少可用 `DAY0`）：

```
DATASET5/
  data/
    DAY0/
      images/        # 原始细胞图像（JPG/PNG/TIF）
      processed/     # ImageJ等价预处理/背景扣除的输出（脚本生成）
      masks/         # 细胞分割输出（脚本生成）
      nuclei/        # 细胞核分割输出（脚本生成）
      features/      # 单图特征CSV（脚本生成）
      total/         # 合并后的CSV（脚本生成）
      overlay_both/  # 可视化叠加（细胞+核）
```

命名匹配规则（很重要）：
- 以“根名”匹配，自动去掉前缀 `processed_` 与后缀 `_mask`/`_masks` 后进行配对。
- 掩码优先使用 `.tif/.tiff`，无则回退 `.png`（仅在可视化阶段）。

## 环境依赖
至少需要：
```
python -m pip install -U opencv-python numpy pandas scikit-image cellpose
```
可选（非必须）：PyImageJ（仅在需要与 ImageJ 完全一致的背景扣除时）：
```
python -m pip install -U pyimagej scyjava
```

## 步骤0：ImageJ 等价预处理（蓝通道→反相→Subtract Background rolling=10）
使用 `DATASET5\code\macro_py.py`（与宏 `macro.ijm` 等价），将 `DAY*/images` 处理到 `DAY*/processed`。

- 处理单个 DAY（推荐 imagej 后端，数值与 ImageJ 一致）：
```
python DATASET5\code\macro_py.py -i DATASET5\data\DAY0\images -o DATASET5\data\DAY0\processed -r 10 --backend imagej --workers 1
```
- 遍历所有 DAY（按数字排序）：
```
Get-ChildItem "DATASET5\data" -Directory -Filter "DAY*" |
  Sort-Object { [int]($_.Name -replace '^\D+','') } |
  ForEach-Object {
    python DATASET5\code\macro_py.py -i "$($_.FullName)\images" -o "$($_.FullName)\processed" -r 10 --backend imagej --workers 1
  }
```
说明：
- `--backend imagej` 最准确但单线程；如需更快可用 `--backend skimage` 或 `--backend morph` 并配合 `--workers auto`。
- 也可用并行启动器（多进程分片）：
```
python DATASET5\code\launch_parallel.py -i DATASET5\data\DAY0\images -o DATASET5\data\DAY0\processed -r 10 --backend imagej --shards 4
```

## 步骤一：细胞分割（images -> masks）
```
python DATASET5\code\1seg_triangle_V6.py --data-root DATASET5\data --mode cells
```
输出：`DAY*/masks`（包含 `*_masks.tif` 与可选 `*_mask.png`）。

## 步骤二：细胞核分割（processed -> nuclei）
核来源于已处理图像（无需再次做蓝通道/反相/背景扣除预处理）：
```
python DATASET5\code\1seg_triangle_V6.py --data-root DATASET5\data --mode nuclei --input-subdir processed --output-subdir nuclei
```
输出：`DAY*/nuclei`（优先 `.tif/.tiff`）。

## 步骤三：特征提取（含核/细胞面积占比与核RGB）
仅使用 `masks/` 与 `nuclei/` 中的 TIF/TIFF 掩码；默认原图用于核RGB统计来自 `images/`：
```
python DATASET5\code\2featureExtract_circularity_V6.py \
  --data-root DATASET5\data \
  --nuclei-subfolder nuclei \
  --rgb-subfolder images \
  --columns minimal
```
- 最小列集 `minimal` 包含：`image,label,area,perimeter,major_axis_length,minor_axis_length,aspect_ratio,eccentricity,orientation_deg,centroid_x,centroid_y,nuclear_area,nuclear_fraction,nuclear_mean_R,nuclear_mean_G,nuclear_mean_B`
- 若你的核或原图不在默认子目录，可改 `--nuclei-subfolder` / `--rgb-subfolder`。
- 运行时会打印：每个 Day 找到多少掩码、配到多少核掩码，便于诊断配对问题。

## 步骤四：按 Day 合并 CSV
```
python DATASET5\code\3combineCsv_V6.py --data-root DATASET5\data
```
输出：每个 `DAY*` 目录下生成 `total\merged.csv`。

## 步骤五：可视化叠加
- 单掩码叠加（示例：核到原图）：
```
python DATASET5\code\visualize_mask.py \
  --data-root DATASET5\data \
  --mask-subfolder nuclei \
  --overlay-subfolder overlay \
  --alpha 0.5
```
- 细胞+核同时叠加（不同颜色区分）：
```
python DATASET5\code\visualize_mask.py \
  --data-root DATASET5\data \
  --cell-subfolder masks \
  --nuclei-subfolder nuclei \
  --overlay-subfolder overlay_both \
  --alpha 0.5 \
  --cell-color 0,255,0 \
  --nuclei-color 0,0,255 \
  --overlap-color 0,255,255
```

## 常见问题与排查
- merged.csv 没有 `nuclear_area/nuclear_fraction`：
  - 检查特征提取时日志中“匹配到 X 份核掩码”；若为 0，说明 `nuclei` 命名或后缀不匹配。
  - 确保 `nuclei` 使用 TIF/TIFF（脚本默认仅扫描 TIF/TIFF）。
- 运行慢：
  - 首先减少控制台输出；或分 Day 运行。
  - 如仍需提速，可告知，我可加入开关关闭 `max_diameter` 计算并调大并行 `chunksize`。

## 命名与匹配规则（再次强调）
- 根名 = 去 `processed_` 前缀 + 去 `_mask/_masks` 后缀；按根名在 `images/masks/nuclei` 间配对。
- 优先 `.tif/.tiff`，无则回退 `.png`（仅可视化阶段）。

## 版本与兼容性提示
- Cellpose 新版本会提示 “model_type argument is not used …” 为正常日志，不影响运行。
- Windows 下并行时如遇卡顿，可先单进程验证，再按 Day 分批处理。

---
若目录名或文件后缀与你的实际数据不一致，可告诉我，我会在脚本中专门适配你的结构与命名。
