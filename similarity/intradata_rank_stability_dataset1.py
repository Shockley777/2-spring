import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, entropy, wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import tempfile


# 基本参数（DATASET1）
DATASET1_BASE = r"D:\project\2-spring\DATASET1\data"
AREA_COL = "area"
BINS = 50
RANGE = (500, 3500)
DAYS = [f"DAY{i}" for i in range(1, 7)]
DATA_FOLDERS = [f"data{j}" for j in range(1, 6)]
RESULT_DIR = os.path.join("similarity", "results", "rank_stability_dataset1")

# 稳定性来源：'vs_full'（推荐）或 'from_excel'
STABILITY_MODE = 'vs_full'
STABILITY_XLSX = r"D:\project\2-spring\DATASET1\code\stability\dataset1_stability_analysis.xlsx"

# vs_full 估计参数（与全量曲线 IoU）
STAB_THRESHOLD = 0.98
STAB_STEP = 250
STAB_CONSECUTIVE = 3
STAB_REPEATS = 3

# 轨迹评估参数
INITIAL_N = 500              # 初始样本量
TRACK_STEP = 250             # 轨迹步长
TRACK_REPEATS = 3            # 每个 N 重复抽样，取平均相似度
SIM_METHOD = 'intersection'  # intersection/cosine/pearson/chi_square/kl/wasserstein/iou


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def get_writable_output_dir(preferred_dir: str, min_free_mb: int = 50) -> str:
    env_dir = os.environ.get("RANK_STAB_DS1_OUTDIR")
    candidates = []
    if env_dir:
        candidates.append(env_dir)
    candidates.append(preferred_dir)
    candidates.append(os.path.join(tempfile.gettempdir(), "rank_stability_dataset1"))

    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            total, used, free = shutil.disk_usage(path)
            if free >= min_free_mb * 1024 * 1024:
                return path
        except Exception:
            continue

    fallback = candidates[-1]
    try:
        os.makedirs(fallback, exist_ok=True)
    except Exception:
        pass
    print("⚠️ 输出目录空间可能不足，将尝试写入临时目录。")
    return fallback


def load_area_series(base_dir: str, day_folder: str, data_folder: str) -> np.ndarray | None:
    csv_path = os.path.join(base_dir, day_folder, data_folder, "total", "merged.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if AREA_COL not in df.columns:
        return None
    area = df[AREA_COL].dropna().values
    area = area[(area >= RANGE[0]) & (area <= RANGE[1])]
    if area.size < 5:
        return None
    return area


def build_hist_from_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(RANGE[0], RANGE[1], BINS + 1)
    hist, _ = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return hist, centers


def build_hist_from_sample(area_values: np.ndarray, sample_size: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if area_values.size < sample_size:
        return None, None
    idx = np.random.choice(area_values.size, size=sample_size, replace=False)
    sampled = area_values[idx]
    return build_hist_from_values(sampled)


def compute_similarity(hist1: np.ndarray, hist2: np.ndarray, method: str, bin_centers: np.ndarray) -> float:
    h1 = np.asarray(hist1, dtype=np.float64) + 1e-10
    h2 = np.asarray(hist2, dtype=np.float64) + 1e-10
    h1 /= np.sum(h1)
    h2 /= np.sum(h2)
    if method == 'intersection':
        return float(np.sum(np.minimum(h1, h2)))
    elif method == 'cosine':
        return float(cosine_similarity(h1.reshape(1, -1), h2.reshape(1, -1))[0, 0])
    elif method == 'pearson':
        corr, _ = pearsonr(h1, h2)
        return float(0 if np.isnan(corr) else corr)
    elif method == 'chi_square':
        denom = h1 + h2
        return float(0.5 * np.sum(((h1 - h2) ** 2) / denom))
    elif method == 'kl':
        return float(entropy(h1, h2))
    elif method == 'wasserstein':
        return float(wasserstein_distance(bin_centers, bin_centers, h1, h2))
    elif method == 'iou':
        return float(np.sum(np.minimum(h1, h2)) / np.sum(np.maximum(h1, h2)))
    else:
        raise ValueError(f"Unsupported method: {method}")


def load_stability_from_excel(xlsx_path: str) -> dict[str, int]:
    if not os.path.exists(xlsx_path):
        return {}
    try:
        df = pd.read_excel(xlsx_path, sheet_name='详细结果')
    except Exception:
        return {}

    def path_to_key(p: str) -> str | None:
        try:
            parts = os.path.normpath(p).split(os.sep)
            if len(parts) >= 4:
                return f"{parts[-4]}_{parts[-3]}"  # DAYx_datay
            return None
        except Exception:
            return None

    stab: dict[str, int] = {}
    for _, row in df.iterrows():
        key = path_to_key(str(row.get('file_path', '')))
        val = row.get('stable_sample_size', None)
        if key and pd.notna(val) and int(val) > 0:
            stab[key] = int(val)
    return stab


def estimate_stable_size_vs_full(area_values: np.ndarray) -> int | None:
    if area_values.size < STAB_STEP * 2:
        return None
    bins = np.linspace(RANGE[0], RANGE[1], BINS + 1)
    full_hist, _ = np.histogram(area_values, bins=bins, density=True)
    sims: list[float] = []
    ns: list[int] = []
    for n in range(STAB_STEP, area_values.size + 1, STAB_STEP):
        vals = []
        for _ in range(STAB_REPEATS):
            idx = np.random.choice(area_values.size, size=n, replace=False)
            sampled = area_values[idx]
            hist, _ = np.histogram(sampled, bins=bins, density=True)
            iou = np.sum(np.minimum(hist, full_hist)) / (np.sum(np.maximum(hist, full_hist)) + 1e-10)
            vals.append(iou)
        sims.append(float(np.mean(vals)))
        ns.append(n)
        if len(sims) >= STAB_CONSECUTIVE and all(x >= STAB_THRESHOLD for x in sims[-STAB_CONSECUTIVE:]):
            return ns[-STAB_CONSECUTIVE]
    if sims:
        return ns[int(np.argmax(sims))]
    return None


def main():
    np.random.seed(42)
    out_dir = get_writable_output_dir(RESULT_DIR, min_free_mb=50)
    ensure_dir(out_dir)

    # 预构建所有目标组的“全量直方图”
    target_hists: dict[str, np.ndarray] = {}
    bin_centers = None
    for day in DAYS:
        for dfolder in DATA_FOLDERS:
            key = f"{day}_{dfolder}"
            area = load_area_series(DATASET1_BASE, day, dfolder)
            if area is None:
                continue
            hist, centers = build_hist_from_values(area)
            target_hists[key] = hist
            bin_centers = centers

    if not target_hists:
        print("❌ 未找到任何目标直方图")
        return

    # 如需 from_excel
    stab_map = load_stability_from_excel(STABILITY_XLSX) if STABILITY_MODE == 'from_excel' else {}

    summary_rows = []

    # 对每个参考组做轨迹分析
    for day in DAYS:
        for dfolder in DATA_FOLDERS:
            ref_key = f"{day}_{dfolder}"
            area = load_area_series(DATASET1_BASE, day, dfolder)
            if area is None:
                print(f"⚠️ 跳过 {ref_key}")
                continue

            # 确定稳定样本量
            if STABILITY_MODE == 'from_excel':
                stable_n = stab_map.get(ref_key, None)
            else:
                stable_n = estimate_stable_size_vs_full(area)
            if stable_n is None:
                stable_n = min(max(area.size // 2, INITIAL_N), area.size)
                print(f"   → {ref_key} 未得稳定点，回退 stable_n={stable_n}")

            # 轨迹 N 列表
            n_list = list(range(INITIAL_N, min(stable_n, area.size) + 1, TRACK_STEP))
            if not n_list:
                n_list = [min(area.size, stable_n)]

            traj = []  # (N, top1_key, top1_score)
            for n in n_list:
                # 计算 ref(n) vs all targets 的平均相似度
                scores = {k: 0.0 for k in target_hists.keys()}
                for _ in range(TRACK_REPEATS):
                    ref_hist, centers = build_hist_from_sample(area, n)
                    # 可能由于极限 n 不足返回 None
                    if ref_hist is None:
                        continue
                    for tgt_key, tgt_hist in target_hists.items():
                        scores[tgt_key] += compute_similarity(ref_hist, tgt_hist, SIM_METHOD, bin_centers if bin_centers is not None else centers)
                # 平均
                for k in scores.keys():
                    scores[k] /= max(1, TRACK_REPEATS)

                # 选择 Top1（相似度越大越好；若使用距离型，可在此调整方向）
                top1_key = max(scores.items(), key=lambda x: x[1])[0]
                top1_score = scores[top1_key]
                traj.append((n, top1_key, top1_score))

            # 判断 Top1 是否始终不变
            first_top1 = traj[0][1]
            switches = sum(1 for i in range(1, len(traj)) if traj[i][1] != traj[i-1][1])
            always_first = all(t[1] == first_top1 for t in traj)

            # 保存轨迹 CSV
            df_traj = pd.DataFrame(traj, columns=["N", "Top1", f"Top1_{SIM_METHOD}"])
            csv_path = os.path.join(out_dir, f"rank_traj_{ref_key}.csv")
            try:
                df_traj.to_csv(csv_path, index=False)
            except OSError as e:
                print(f"⚠️ 保存CSV失败：{e}")

            # 绘图：Top1 分数随 N 变化；并标记 Top1 是否更换
            try:
                plt.figure(figsize=(10, 5))
                plt.plot(df_traj["N"], df_traj[f"Top1_{SIM_METHOD}"], marker='o')
                for i in range(1, len(traj)):
                    if traj[i][1] != traj[i-1][1]:
                        plt.axvline(traj[i][0], color='red', linestyle='--', alpha=0.5)
                plt.title(f"Top1 Score Trajectory | Ref={ref_key} | method={SIM_METHOD}\nAlways First={always_first}, Switches={switches}")
                plt.xlabel("Sample Size N")
                plt.ylabel(f"Top1 {SIM_METHOD}")
                plt.tight_layout()
                fig_path = os.path.join(out_dir, f"rank_traj_{ref_key}.png")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
            except OSError as e:
                print(f"⚠️ 保存图失败：{e}")

            summary_rows.append({
                "Reference": ref_key,
                "Stable_N": stable_n,
                "Initial_N": n_list[0],
                "Steps": len(n_list),
                "Always_First": always_first,
                "Num_Switches": switches,
                "First_Top1": first_top1,
                "Final_Top1": traj[-1][1],
            })

    # 汇总
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        sum_path = os.path.join(out_dir, "rank_stability_summary.csv")
        try:
            df_sum.to_csv(sum_path, index=False)
        except OSError as e:
            print(f"⚠️ 保存汇总失败：{e}")


if __name__ == "__main__":
    main()



