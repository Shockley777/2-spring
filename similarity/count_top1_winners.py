import os
import glob
import re
from collections import Counter, defaultdict

import pandas as pd


DATASET_DIRS = {
    "intra_dataset1": os.path.join("similarity", "results", "intra_dataset1"),
    "intra_dataset2": os.path.join("similarity", "results", "intra_dataset2"),
    "intra_dataset3": os.path.join("similarity", "results", "intra_dataset3"),
    "intra_dataset4": os.path.join("similarity", "results", "intra_dataset4"),
}

PRIMARY_COL = "Histogram Intersection"
OUTPUT_DIR = os.path.join("similarity", "results", "top1_counts")


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def read_top1_target(xlsx_path: str) -> str | None:
    """从一个excel（Intersection sheet）读取Top-1的目标名（Compared Folder）。"""
    try:
        df = pd.read_excel(xlsx_path, sheet_name="intersection")
    except Exception:
        try:
            df = pd.read_excel(xlsx_path, sheet_name="Intersection")
        except Exception:
            return None

    # 若存在主列名，按其排序；否则退化为按第二列
    if PRIMARY_COL in df.columns:
        df_sorted = df.sort_values(by=PRIMARY_COL, ascending=False)
    else:
        cols = list(df.columns)
        if len(cols) < 2:
            return None
        df_sorted = df.sort_values(by=cols[1], ascending=False)

    if df_sorted.empty:
        return None
    target = df_sorted.iloc[0].get("Compared Folder")
    return str(target) if pd.notna(target) else None


def count_winners() -> dict:
    results: dict[str, Counter] = {}
    for key, dir_path in DATASET_DIRS.items():
        if not os.path.isdir(dir_path):
            continue
        c = Counter()
        for xlsx in glob.glob(os.path.join(dir_path, "*.xlsx")):
            top1 = read_top1_target(xlsx)
            if top1:
                c[top1] += 1
        results[key] = c
    return results


def save_reports(results: dict) -> None:
    ensure_dir(OUTPUT_DIR)
    # 各数据集分别导出
    for ds_key, counter in results.items():
        rows = [(name, count) for name, count in counter.most_common()]
        df = pd.DataFrame(rows, columns=["Target", "Top1_Count"]) if rows else pd.DataFrame(columns=["Target", "Top1_Count"])
        out_csv = os.path.join(OUTPUT_DIR, f"{ds_key}_top1_counts.csv")
        df.to_csv(out_csv, index=False)
    # 合并导出
    merged = defaultdict(int)
    for counter in results.values():
        for name, count in counter.items():
            merged[name] += count
    rows = [(name, count) for name, count in sorted(merged.items(), key=lambda x: x[1], reverse=True)]
    df_all = pd.DataFrame(rows, columns=["Target", "Top1_Count_All_DS"]) if rows else pd.DataFrame(columns=["Target", "Top1_Count_All_DS"])
    df_all.to_csv(os.path.join(OUTPUT_DIR, "all_datasets_top1_counts.csv"), index=False)


def main():
    results = count_winners()
    save_reports(results)
    # 控制台简报
    for ds_key, counter in results.items():
        total_refs = sum(1 for _ in glob.glob(os.path.join(DATASET_DIRS.get(ds_key, ""), "*.xlsx")))
        unique_winners = len(counter)
        print(f"[{ds_key}] 参考组数: {total_refs}, 能排第一的目标(去重): {unique_winners}")
        for name, cnt in counter.most_common(5):
            print(f"  - {name}: {cnt}")
    print(f"\nCSV已导出到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


