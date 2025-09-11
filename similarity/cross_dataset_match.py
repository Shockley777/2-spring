import os
import numpy as np
import pandas as pd
import shutil
import tempfile


# 全局参数
DATASET1_BASE = r"D:\project\2-spring\DATASET1\data"
DATASET2_BASE = r"D:\project\2-spring\DATASET2\data"
AREA_COL = "area"
BINS = 50
RANGE = (500, 3500)
REF_RATIO = 1.0  # 参考库使用全部数据
TARGET_RATIO = 0.8  # 目标曲线抽样比例，可按需调整
TOP_K = 5
RESULT_DIR = os.path.join("similarity", "results", "cross_dataset")

# 可选：手动指定目标曲线键，例如 'DS2::DAY4_data1'；为 None 时随机选择
TARGET_KEY = None

# 仅输出直方图交集的 Top1 结果为 CSV 合集

# 仅使用直方图交集作为相似度指标（越大越好）


def get_writable_output_dir(preferred_dir: str, min_free_mb: int = 100) -> str:
    """选择可写且有足够剩余空间的输出目录。

    选择顺序：
    1) 环境变量 CROSS_DATASET_OUTDIR 指定路径（若有足够空间）
    2) preferred_dir（若有足够空间）
    3) 系统临时目录下的 cross_dataset 子目录
    若都不足，则仍返回临时目录（但会打印警告）。
    """
    # 1) 环境变量优先
    env_dir = os.environ.get("CROSS_DATASET_OUTDIR")
    candidates = []
    if env_dir:
        candidates.append(env_dir)
    candidates.append(preferred_dir)
    candidates.append(os.path.join(tempfile.gettempdir(), "cross_dataset"))

    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            total, used, free = shutil.disk_usage(path)
            if free >= min_free_mb * 1024 * 1024:
                return path
        except Exception:
            continue

    # 所有候选都不足，返回最后一个并提示
    fallback = candidates[-1]
    try:
        os.makedirs(fallback, exist_ok=True)
    except Exception:
        pass
    print("⚠️ 所有输出目录空间可能不足，将尝试写入临时目录，可能仍会失败。")
    return fallback


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def load_histogram(base_dir: str, day_folder: str, data_folder: str, ratio: float):
    csv_path = os.path.join(base_dir, day_folder, data_folder, "total", "merged.csv")
    if not os.path.exists(csv_path):
        return None, None

    df = pd.read_csv(csv_path)
    if AREA_COL not in df.columns:
        return None, None

    area_series = df[AREA_COL].dropna().values
    area_series = area_series[(area_series >= RANGE[0]) & (area_series <= RANGE[1])]
    if area_series.size < 5:
        return None, None

    np.random.shuffle(area_series)
    sampled = area_series[: int(area_series.size * ratio)] if 0 < ratio < 1 else area_series

    bins = np.linspace(RANGE[0], RANGE[1], BINS + 1)
    hist, _ = np.histogram(sampled, bins=bins, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return hist, bin_centers


def compute_histogram_intersection(hist1: np.ndarray, hist2: np.ndarray) -> float:
    h1 = np.asarray(hist1, dtype=np.float64) + 1e-10
    h2 = np.asarray(hist2, dtype=np.float64) + 1e-10
    h1 /= np.sum(h1)
    h2 /= np.sum(h2)
    return float(np.sum(np.minimum(h1, h2)))


def compute_similarity_intersection_only(hist1: np.ndarray, hist2: np.ndarray) -> float:
    h1 = np.asarray(hist1, dtype=np.float64) + 1e-10
    h2 = np.asarray(hist2, dtype=np.float64) + 1e-10
    h1 /= np.sum(h1)
    h2 /= np.sum(h2)
    return float(np.sum(np.minimum(h1, h2)))


def build_reference_library() -> dict:
    days_ds1 = [f"DAY{i}" for i in range(1, 7)]
    data_folders_ds1 = [f"data{j}" for j in range(1, 6)]

    reference_histograms = {}
    for day in days_ds1:
        for data_folder in data_folders_ds1:
            key = f"DS1::{day}_{data_folder}"
            hist, bins = load_histogram(DATASET1_BASE, day, data_folder, REF_RATIO)
            if hist is not None:
                reference_histograms[key] = hist
    return reference_histograms


def gather_dataset2_targets() -> dict:
    days_ds2 = [f"DAY{i}" for i in range(2, 8)]
    data_folders_ds2 = [f"data{j}" for j in range(1, 7)]

    target_histograms = {}
    for day in days_ds2:
        for data_folder in data_folders_ds2:
            key = f"DS2::{day}_{data_folder}"
            hist, bins = load_histogram(DATASET2_BASE, day, data_folder, TARGET_RATIO)
            if hist is not None:
                target_histograms[key] = hist
    return target_histograms


def main():
    # 选择输出目录：优先使用环境变量/既定目录，空间不足则自动回退到临时目录
    out_dir = get_writable_output_dir(RESULT_DIR, min_free_mb=50)
    ensure_dir(out_dir)

    # 1) 构建 DATASET1 参考库
    reference_histograms = build_reference_library()
    if not reference_histograms:
        print("❌ 未能从 DATASET1 构建任何参考直方图，请检查数据路径与文件。")
        return
    print(f"✅ 参考库载入成功：{len(reference_histograms)} 条曲线")

    # 2) 收集 DATASET2 所有可作为目标的曲线
    target_histograms = gather_dataset2_targets()
    if not target_histograms:
        print("❌ 未能从 DATASET2 载入任何目标直方图，请检查数据路径与文件。")
        return
    print(f"✅ 目标曲线载入成功：{len(target_histograms)} 条曲线")

    # 3) 对每个目标计算与参考库的直方图交集 Top1
    results = []
    for target_key, target_hist in target_histograms.items():
        best_ref = None
        best_score = -1.0
        for ref_key, ref_hist in reference_histograms.items():
            score = compute_similarity_intersection_only(target_hist, ref_hist)
            if score > best_score:
                best_score = score
                best_ref = ref_key
        if best_ref is not None:
            results.append({
                "Target": target_key,
                "BestReference": best_ref,
                "Intersection": best_score,
            })

    if not results:
        print("❌ 未得到任何匹配结果。")
        return

    result_df = pd.DataFrame(results)
    # 4) 写出单个 CSV 合集
    csv_path = os.path.join(out_dir, "ds2_to_ds1_intersection_top1.csv")
    result_df.sort_values(by=["Intersection"], ascending=False).to_csv(csv_path, index=False)
    print(f"\n📁 已输出 Top1 合集：{csv_path}")


if __name__ == "__main__":
    main()


