import os
import cv2
import numpy as np
import argparse
import re


def parse_color(color_str: str) -> tuple:
	"""解析颜色字符串：
	- 形如 "B,G,R"（0-255）
	- 或十六进制 "#RRGGBB"（会转换为BGR）
	默认返回绿色或红色等。
	"""
	if not color_str:
		return (0, 255, 0)
	s = color_str.strip()
	if s.startswith('#') and len(s) == 7:
		# #RRGGBB -> BGR
		r = int(s[1:3], 16)
		g = int(s[3:5], 16)
		b = int(s[5:7], 16)
		return (b, g, r)
	if ',' in s:
		parts = s.split(',')
		if len(parts) == 3:
			b, g, r = [int(p.strip()) for p in parts]
			return (b, g, r)
	# fallback 绿色
	return (0, 255, 0)


def overlay_two_masks_on_image(image_path, cell_mask_path, nuclei_mask_path, output_path,
							   alpha=0.5, cell_color=(0,255,0), nuclei_color=(0,0,255), overlap_color=(0,255,255)):
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	if image is None:
		print(f"无法读取图像: {image_path}")
		return False

	def load_mask(p):
		if not p:
			return None
		m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
		if m is None:
			return None
		if m.ndim == 3:
			m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
		return (m > 0).astype(np.uint8)

	cell_bin = load_mask(cell_mask_path)
	nuc_bin = load_mask(nuclei_mask_path)

	if cell_bin is None and nuc_bin is None:
		print("无可用掩码，跳过: " + image_path)
		return False

	if cell_bin is None:
		cell_bin = np.zeros(image.shape[:2], dtype=np.uint8)
	if nuc_bin is None:
		nuc_bin = np.zeros(image.shape[:2], dtype=np.uint8)

	combined = np.zeros_like(image)
	# 仅细胞
	combined[cell_bin.astype(bool) & ~nuc_bin.astype(bool)] = cell_color
	# 仅核
	combined[nuc_bin.astype(bool) & ~cell_bin.astype(bool)] = nuclei_color
	# 重叠（核∩细胞）
	combined[(cell_bin & nuc_bin).astype(bool)] = overlap_color

	overlay = cv2.addWeighted(image, 1.0, combined, float(alpha), 0)
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	cv2.imwrite(output_path, overlay)
	return True


def root_key(name: str) -> str:
	base = os.path.splitext(name)[0]
	base = re.sub(r'^(processed_)+', '', base, flags=re.IGNORECASE)
	base = re.sub(r'_(mask|masks)$', '', base, flags=re.IGNORECASE)
	return base.lower()


def build_mask_index(mask_folder: str) -> dict:
	index = {}
	if not os.path.isdir(mask_folder):
		return index
	# 仅使用 tif/tiff；若没有再回退 png
	tif_first = {}
	png_fallback = {}
	for fn in os.listdir(mask_folder):
		path = os.path.join(mask_folder, fn)
		if not os.path.isfile(path):
			continue
		ext = os.path.splitext(fn)[1].lower()
		rk = root_key(fn)
		if ext in ('.tif', '.tiff'):
			tif_first[rk] = path
		elif ext in ('.png',):
			png_fallback[rk] = path
	# 合并：优先 tif，其次 png
	for rk, p in tif_first.items():
		index[rk] = p
	for rk, p in png_fallback.items():
		index.setdefault(rk, p)
	return index


def process_day_folder(day_path: str, cell_subfolder: str, nuclei_subfolder: str, overlay_subfolder: str,
					   alpha: float, cell_color, nuclei_color, overlap_color) -> None:
	images_dir = os.path.join(day_path, 'images')
	cell_dir = os.path.join(day_path, cell_subfolder) if cell_subfolder else None
	nuc_dir = os.path.join(day_path, nuclei_subfolder) if nuclei_subfolder else None
	overlay_dir = os.path.join(day_path, overlay_subfolder)
	if not os.path.isdir(images_dir):
		print(f"跳过（无 images）: {day_path}")
		return
	cell_index = build_mask_index(cell_dir) if cell_dir else {}
	nuc_index = build_mask_index(nuc_dir) if nuc_dir else {}
	if not cell_index and not nuc_index:
		print(f"警告：{day_path} 未找到任何掩码（{cell_subfolder} / {nuclei_subfolder}）")

	image_files = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')]
	done, miss = 0, 0
	for fn in image_files:
		rk = root_key(fn)
		cell_path = cell_index.get(rk)
		nuc_path = nuc_index.get(rk)
		if not cell_path and not nuc_path:
			miss += 1
			continue
		image_path = os.path.join(images_dir, fn)
		out_path = os.path.join(overlay_dir, f"{os.path.splitext(fn)[0]}_overlay.png")
		ok = overlay_two_masks_on_image(image_path, cell_path, nuc_path, out_path, alpha, cell_color, nuclei_color, overlap_color)
		if ok:
			done += 1
	print(f"{os.path.basename(day_path)}: 成功 {done}，未匹配 {miss}（cells={cell_subfolder}, nuclei={nuclei_subfolder}）")


def process_all_days(data_root: str, cell_subfolder: str, nuclei_subfolder: str, overlay_subfolder: str,
					 alpha: float, cell_color, nuclei_color, overlap_color) -> None:
	entries = [d for d in sorted(os.listdir(data_root)) if os.path.isdir(os.path.join(data_root, d)) and d.strip().lower().startswith('day')]
	if not entries:
		print(f"未发现 Day*/DAY* 目录于 {data_root}")
		return
	for d in entries:
		process_day_folder(os.path.join(data_root, d), cell_subfolder, nuclei_subfolder, overlay_subfolder,
						 alpha, cell_color, nuclei_color, overlap_color)


def main():
	parser = argparse.ArgumentParser(description='将 DAY*/images 与 DAY*/(masks & nuclei) 同时叠加可视化（不同颜色区分）')
	parser.add_argument('--data-root', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')), help='数据根目录，包含 Day*/DAY*')
	parser.add_argument('--cell-subfolder', type=str, default='masks', help='细胞掩码子目录（默认 masks）')
	parser.add_argument('--nuclei-subfolder', type=str, default='nuclei', help='核掩码子目录（默认 nuclei）')
	parser.add_argument('--overlay-subfolder', type=str, default='overlay_both', help='输出叠加图子目录名')
	parser.add_argument('--alpha', type=float, default=0.5, help='叠加透明度 [0,1]')
	parser.add_argument('--cell-color', type=str, default='0,255,0', help='细胞颜色 B,G,R 或 #RRGGBB，默认绿色')
	parser.add_argument('--nuclei-color', type=str, default='0,0,255', help='核颜色 B,G,R 或 #RRGGBB，默认红色')
	parser.add_argument('--overlap-color', type=str, default='0,255,255', help='重叠颜色 B,G,R 或 #RRGGBB，默认黄色')
	args = parser.parse_args()

	cell_color = parse_color(args.cell_color)
	nuclei_color = parse_color(args.nuclei_color)
	overlap_color = parse_color(args.overlap_color)

	process_all_days(args.data_root, args.cell_subfolder, args.nuclei_subfolder, args.overlay_subfolder,
					 args.alpha, cell_color, nuclei_color, overlap_color)
	print("\n所有图像处理完成！")


if __name__ == '__main__':
	main() 