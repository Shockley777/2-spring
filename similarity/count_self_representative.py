import os
import glob
import re
import pandas as pd


DATASET_DIRS = {
    "intra_dataset1": os.path.join("similarity", "results", "intra_dataset1"),
    "intra_dataset2": os.path.join("similarity", "results", "intra_dataset2"),
    "intra_dataset3": os.path.join("similarity", "results", "intra_dataset3"),
    "intra_dataset4": os.path.join("similarity", "results", "intra_dataset4"),
}

PRIMARY_COL = "Histogram Intersection"
OUTPUT_DIR = os.path.join("similarity", "results", "self_representative")


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def parse_reference_from_filename(path: str) -> str:
    # intra_dsX_<REF>_N####.xlsx -> <REF>
    base = os.path.basename(path)
    m = re.search(r"intra_ds\d+_(.+?)_N\d+\.xlsx", base, re.IGNORECASE)
    return m.group(1) if m else base


def read_top1_target(xlsx_path: str) -> str | None:
    try:
        df = pd.read_excel(xlsx_path, sheet_name="intersection")
    except Exception:
        try:
            df = pd.read_excel(xlsx_path, sheet_name="Intersection")
        except Exception:
            return None

    if df.empty:
        return None

    if PRIMARY_COL in df.columns:
        df = df.sort_values(by=PRIMARY_COL, ascending=False)
    else:
        cols = list(df.columns)
        if len(cols) >= 2:
            df = df.sort_values(by=cols[1], ascending=False)
        else:
            return None

    top_folder = df.iloc[0].get("Compared Folder")
    return str(top_folder) if pd.notna(top_folder) else None


def evaluate_dataset(ds_key: str, dir_path: str) -> pd.DataFrame:
    rows = []
    for xlsx in glob.glob(os.path.join(dir_path, "*.xlsx")):
        ref = parse_reference_from_filename(xlsx)
        top1 = read_top1_target(xlsx)
        if not top1:
            continue
        rows.append({
            "Dataset": ds_key,
            "Reference": ref,
            "Top1": top1,
            "Is_Self_First": ref == top1,
        })
    return pd.DataFrame(rows)


def main():
    ensure_dir(OUTPUT_DIR)
    all_rows = []
    for ds_key, dir_path in DATASET_DIRS.items():
        if not os.path.isdir(dir_path):
            continue
        df = evaluate_dataset(ds_key, dir_path)
        all_rows.append(df)
        # 单数据集报告
        if not df.empty:
            total = len(df)
            self_first = int(df["Is_Self_First"].sum())
            ratio = self_first / total if total else 0.0
            print(f"[{ds_key}] 参考组数={total}, 自身排第一={self_first} ({ratio:.1%})")
            df.to_csv(os.path.join(OUTPUT_DIR, f"{ds_key}_self_first_detail.csv"), index=False)

    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
        df_all.to_csv(os.path.join(OUTPUT_DIR, "all_self_first_detail.csv"), index=False)
        summary = df_all.groupby("Dataset")["Is_Self_First"].agg(["sum", "count"]).reset_index()
        summary["ratio"] = summary["sum"] / summary["count"]
        summary.columns = ["Dataset", "Self_First", "Total", "Ratio"]
        summary.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
        print("\n汇总:")
        for r in summary.itertuples(index=False):
            print(f"- {r.Dataset}: 自身排第一 {int(r.Self_First)}/{int(r.Total)} ({r.Ratio:.1%})")
    else:
        print("未找到任何 intra_datasetX 的结果文件。")


if __name__ == "__main__":
    main()



