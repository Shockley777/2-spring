import os
import csv
from typing import List, Dict, Tuple


AREA_MIN = 500.0
AREA_MAX = 3500.0


def count_images_like(dirpath: str) -> int:
    """Count image files. Prefer 'images', then 'data', else fall back to 'features' csv count."""
    images_dir = os.path.join(dirpath, "images")
    data_dir = os.path.join(dirpath, "data")
    features_dir = os.path.join(dirpath, "features")

    if os.path.isdir(images_dir):
        return sum(1 for n in os.listdir(images_dir) if n.lower().endswith((".jpg", ".jpeg", ".png")))
    if os.path.isdir(data_dir):
        return sum(1 for n in os.listdir(data_dir) if n.lower().endswith((".jpg", ".jpeg", ".png")))
    if os.path.isdir(features_dir):
        return sum(1 for n in os.listdir(features_dir) if n.lower().endswith(".csv"))
    return 0


def get_area_key(header: List[str]) -> str:
    for key in header or []:
        if key.lower() == "area":
            return key
    raise KeyError("'area' column not found in merged.csv header: %s" % header)


def count_cells_in_csv(csv_path: str, area_min: float, area_max: float) -> Tuple[int, int]:
    total_cells = 0
    cells_in_range = 0

    def _scan(fh) -> Tuple[int, int, str]:
        nonlocal total_cells, cells_in_range
        reader = csv.DictReader(fh)
        area_key = get_area_key(reader.fieldnames or [])
        for row in reader:
            total_cells += 1
            try:
                area_val = float(row.get(area_key, "") or 0)
            except ValueError:
                continue
            if area_min <= area_val <= area_max:
                cells_in_range += 1
        return total_cells, cells_in_range, area_key

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            _scan(f)
    except UnicodeDecodeError:
        with open(csv_path, "r", encoding="gbk", errors="ignore") as f:
            _scan(f)

    return total_cells, cells_in_range


def find_units(base_dir: str) -> List[str]:
    units: List[str] = []
    for dirpath, dirnames, filenames in os.walk(base_dir):
        if os.path.basename(dirpath) in {"total", "features", "images", "data", "masks"}:
            # only consider leaf containers by evaluating parent
            parent = os.path.dirname(dirpath)
            merged_csv = os.path.join(parent, "total", "merged.csv")
            has_media = any(
                os.path.isdir(os.path.join(parent, name))
                for name in ("images", "data", "features")
            )
            if has_media and os.path.isfile(merged_csv) and parent not in units:
                units.append(parent)
        else:
            merged_csv = os.path.join(dirpath, "total", "merged.csv")
            has_media = any(
                os.path.isdir(os.path.join(dirpath, name))
                for name in ("images", "data", "features")
            )
            if has_media and os.path.isfile(merged_csv):
                units.append(dirpath)
    return sorted(set(units))


def write_markdown(out_path: str, rows: List[Dict[str, str]]) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("| 文件夹 | 图片数 | 细胞数（所有） | 细胞数（500-3500） |\n")
        f.write("| --- | ---: | ---: | ---: |\n")
        for r in rows:
            f.write(
                f"| {r['folder']} | {r['images_count']} | {r['cells_count']} | {r['cells_in_500_3500']} |\n"
            )


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    units = find_units(base_dir)
    out_rows: List[Dict[str, str]] = []

    for unit in units:
        merged_csv = os.path.join(unit, "total", "merged.csv")
        images_count = count_images_like(unit)
        total_cells, cells_in_range = count_cells_in_csv(merged_csv, AREA_MIN, AREA_MAX)
        rel = os.path.relpath(unit, base_dir).replace("\\", "/")
        out_rows.append(
            {
                "folder": rel,
                "images_count": str(images_count),
                "cells_count": str(total_cells),
                "cells_in_500_3500": str(cells_in_range),
            }
        )

    # sort by folder for stable output
    out_rows.sort(key=lambda r: r["folder"])

    csv_path = os.path.join(base_dir, "processed_counts.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["folder", "images_count", "cells_count", "cells_in_500_3500"]
        )
        writer.writeheader()
        writer.writerows(out_rows)

    md_path = os.path.join(base_dir, "processed_counts.md")
    write_markdown(md_path, out_rows)

    print("Saved:", os.path.relpath(csv_path, start=base_dir))
    print("Saved:", os.path.relpath(md_path, start=base_dir))


if __name__ == "__main__":
    main()


