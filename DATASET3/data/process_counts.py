import os
import csv
from typing import List, Dict, Tuple


def count_images(images_dir: str) -> int:
    if not os.path.isdir(images_dir):
        return 0
    count = 0
    for name in os.listdir(images_dir):
        lower = name.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg"):
            count += 1
    return count


def get_area_key(header: List[str]) -> str:
    for key in header:
        if key.lower() == "area":
            return key
    raise KeyError("'area' column not found in merged.csv header: %s" % header)


def count_cells_in_csv(csv_path: str, area_min: float, area_max: float) -> Tuple[int, int]:
    total_cells = 0
    cells_in_range = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        try:
            area_key = get_area_key(reader.fieldnames or [])
        except KeyError:
            # Try gbk as fallback
            f.close()
            with open(csv_path, "r", encoding="gbk", errors="ignore") as f2:
                reader = csv.DictReader(f2)
                area_key = get_area_key(reader.fieldnames or [])
                for row in reader:
                    total_cells += 1
                    try:
                        area_val = float(row.get(area_key, "") or 0)
                    except ValueError:
                        continue
                    if area_min <= area_val <= area_max:
                        cells_in_range += 1
                return total_cells, cells_in_range

        for row in reader:
            total_cells += 1
            try:
                area_val = float(row.get(area_key, "") or 0)
            except ValueError:
                continue
            if area_min <= area_val <= area_max:
                cells_in_range += 1

    return total_cells, cells_in_range


def find_units(base_dir: str) -> List[str]:
    """
    A "unit" directory is any directory containing both:
      - a subdirectory 'images' (with image files)
      - a file 'total/merged.csv'
    Return list of such directory paths.
    """
    units: List[str] = []
    for dirpath, dirnames, filenames in os.walk(base_dir):
        total_dir = os.path.join(dirpath, "total")
        images_dir = os.path.join(dirpath, "images")
        merged_csv = os.path.join(total_dir, "merged.csv")
        if os.path.isdir(images_dir) and os.path.isfile(merged_csv):
            units.append(dirpath)
    return units


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # The base to scan is the DATASET3/data directory containing this script
    scan_root = base_dir

    out_rows: List[Dict[str, str]] = []
    units = find_units(scan_root)
    units.sort()

    for unit_dir in units:
        images_dir = os.path.join(unit_dir, "images")
        merged_csv = os.path.join(unit_dir, "total", "merged.csv")

        images_count = count_images(images_dir)
        total_cells, cells_in_range = count_cells_in_csv(merged_csv, 500.0, 3500.0)

        rel_path = os.path.relpath(unit_dir, scan_root)
        out_rows.append(
            {
                "folder": rel_path.replace("\\", "/"),
                "images_count": str(images_count),
                "cells_count": str(total_cells),
                "cells_in_500_3500": str(cells_in_range),
            }
        )

    output_csv = os.path.join(scan_root, "processed_counts.csv")
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["folder", "images_count", "cells_count", "cells_in_500_3500"]
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print("Saved:", os.path.relpath(output_csv, start=os.getcwd()))


if __name__ == "__main__":
    main()


