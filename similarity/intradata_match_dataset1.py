import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wasserstein_distance, pearsonr, entropy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import shutil
import tempfile
from datetime import datetime


# 基本参数
DATASET1_BASE = r"D:\project\2-spring\DATASET1\data"
AREA_COL = "area"
BINS = 50
RANGE = (500, 3500)
ALL_GROUPS_DAYS = [f"DAY{i}" for i in range(1, 7)]
ALL_GROUPS_DATAFOLDERS = [f"data{j}" for j in range(1, 6)]
SAMPLE_SIZES = [2500, 3000, 3500, 4000, 4500, 5000]  # 固定抽样细胞数，可按需扩展
TOP_K = 10
RESULT_DIR = os.path.join("similarity", "results", "intra_dataset1")

# 是否使用稳定性分析来决定抽样量；以及稳定性文件路径与策略
USE_STABILITY = True
STABILITY_XLSX = r"D:\project\2-spring\DATASET1\code\stability\dataset1_stability_analysis.xlsx"
# 来源模式：'from_excel' | 'vs_full'
STABILITY_MODE = 'vs_full'
# 策略：'per_file' 每个文件用自己的稳定点；'global_75th' / 'global_90th' / 'global_max'
STABILITY_STRATEGY = 'per_file'

# vs_full 稳定性参数（与全量曲线比较）
STAB_THRESHOLD = 0.98
STAB_STEP = 250
STAB_CONSECUTIVE = 3
STAB_REPEATS = 3  # 每个样本量重复抽样以降低方差

# 统一抽样绘制分布图设置
USE_UNIFIED_SAMPLE_SIZE = True
UNIFIED_SAMPLE_SIZE = 4950

# 指标设置
SIMILARITY_METHODS = [
    'intersection',    # 越大越好
    'cosine',          # 越大越好
    'pearson',         # 越大越好
    'chi_square',      # 越小越好
    'kl',              # 越小越好
    'wasserstein',     # 越小越好
]
DISPLAY_NAMES = {
    'intersection': 'Histogram Intersection',
    'cosine': 'Cosine Similarity',
    'pearson': 'Pearson Correlation',
    'chi_square': 'Chi-Square Distance',
    'kl': 'KL Divergence',
    'wasserstein': 'Wasserstein Distance',
}
HIGHER_BETTER = {'intersection', 'cosine', 'pearson'}
PLOT_METHOD = 'intersection'


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def get_writable_output_dir(preferred_dir: str, min_free_mb: int = 50) -> str:
    env_dir = os.environ.get("INTRA_DATASET1_OUTDIR")
    candidates = []
    if env_dir:
        candidates.append(env_dir)
    candidates.append(preferred_dir)
    candidates.append(os.path.join(tempfile.gettempdir(), "intra_dataset1"))

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
    print("⚠️ 所有输出目录空间可能不足，将尝试写入临时目录，可能仍会失败。")
    return fallback


def load_area_series(base_dir: str, day_folder: str, data_folder: str) -> np.ndarray | None:
    csv_path = os.path.join(base_dir, day_folder, data_folder, "total", "merged.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if AREA_COL not in df.columns:
        return None
    area_series = df[AREA_COL].dropna().values
    area_series = area_series[(area_series >= RANGE[0]) & (area_series <= RANGE[1])]
    if area_series.size < 5:
        return None
    return area_series


def build_hist_from_sample(area_values: np.ndarray, sample_size: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if area_values.size < sample_size:
        return None, None
    idx = np.random.choice(area_values.size, size=sample_size, replace=False)
    sampled = area_values[idx]
    bins = np.linspace(RANGE[0], RANGE[1], BINS + 1)
    hist, _ = np.histogram(sampled, bins=bins, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return hist, bin_centers


def build_full_hist(area_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(RANGE[0], RANGE[1], BINS + 1)
    hist, _ = np.histogram(area_values, bins=bins, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return hist, bin_centers


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
    else:
        raise ValueError(f"Unsupported similarity method: {method}")


def load_stability_map(xlsx_path: str) -> tuple[dict[str, int], int] | tuple[dict[str, int], None]:
    """读取稳定性分析Excel，返回：每组的稳定样本数映射、全局75分位推荐值。

    Excel 由 `dataset1_stability_analysis.py` 生成，sheet='详细结果'，含列：
    - file_path: .../DAYx/dataY/total/merged.csv
    - stable_sample_size: 稳定点（可能为NaN）
    """
    if not os.path.exists(xlsx_path):
        print(f"⚠️ 未找到稳定性文件：{xlsx_path}，将使用固定 SAMPLE_SIZES。")
        return {}, None

    try:
        df = pd.read_excel(xlsx_path, sheet_name='详细结果')
    except Exception as e:
        print(f"⚠️ 读取稳定性文件失败：{e}，将使用固定 SAMPLE_SIZES。")
        return {}, None

    def path_to_key(p: str) -> str | None:
        try:
            parts = os.path.normpath(p).split(os.sep)
            # 期望 .../DAYx/dataY/total/merged.csv
            if len(parts) >= 4:
                day = parts[-4]
                data = parts[-3]
                return f"{day}_{data}"
            return None
        except Exception:
            return None

    stability_map: dict[str, int] = {}
    stable_sizes: list[int] = []
    for _, row in df.iterrows():
        fp = str(row.get('file_path', ''))
        key = path_to_key(fp)
        val = row.get('stable_sample_size', None)
        if key and pd.notna(val) and int(val) > 0:
            stability_map[key] = int(val)
            stable_sizes.append(int(val))

    global_75th = int(np.percentile(stable_sizes, 75)) if stable_sizes else None
    return stability_map, global_75th


def compute_hist_iou(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """IoU 风格交集：sum(min)/sum(max)。使用密度直方图。"""
    h1 = np.asarray(hist1, dtype=np.float64) + 1e-10
    h2 = np.asarray(hist2, dtype=np.float64) + 1e-10
    return float(np.sum(np.minimum(h1, h2)) / np.sum(np.maximum(h1, h2)))


def estimate_stable_size_vs_full(area_values: np.ndarray,
                                 threshold: float = STAB_THRESHOLD,
                                 step: int = STAB_STEP,
                                 consecutive: int = STAB_CONSECUTIVE,
                                 repeats: int = STAB_REPEATS) -> int | None:
    """基于与全量曲线的 IoU 相似度，估计稳定样本量。

    - 在每个样本量 n（步长 step）下，重复抽样 repeats 次，与全量直方图比较 IoU，取平均；
    - 返回第一个连续 consecutive 个点均 >= threshold 的最小 n；若找不到则返回 None。
    """
    if area_values.size < step * 2:
        return None
    bins = np.linspace(RANGE[0], RANGE[1], BINS + 1)
    full_hist, _ = np.histogram(area_values, bins=bins, density=True)

    sims: list[float] = []
    ns: list[int] = []
    max_n = area_values.size
    for n in range(step, max_n + 1, step):
        vals = []
        for _ in range(repeats):
            idx = np.random.choice(area_values.size, size=n, replace=False)
            sampled = area_values[idx]
            hist, _ = np.histogram(sampled, bins=bins, density=True)
            vals.append(compute_hist_iou(hist, full_hist))
        sims.append(float(np.mean(vals)))
        ns.append(n)

        # 检查是否已满足连续条件
        if len(sims) >= consecutive and all(x >= threshold for x in sims[-consecutive:]):
            return ns[-consecutive]

    # 尝试返回达到最高相似度的 n（若未达阈值）
    if sims:
        best_idx = int(np.argmax(sims))
        return ns[best_idx]
    return None


def main():
    # 可复现性
    np.random.seed(42)

    out_dir = get_writable_output_dir(RESULT_DIR, min_free_mb=50)
    ensure_dir(out_dir)

    # 预先构建所有组的直方图（作为被比较对象）
    target_histograms: dict[str, np.ndarray] = {}
    for day in ALL_GROUPS_DAYS:
        for data_folder in ALL_GROUPS_DATAFOLDERS:
            key = f"{day}_{data_folder}"
            area = load_area_series(DATASET1_BASE, day, data_folder)
            if area is None:
                continue
            if USE_UNIFIED_SAMPLE_SIZE:
                if area.size < UNIFIED_SAMPLE_SIZE:
                    # 不足 4950 的组不参与绘制/对比
                    continue
                hist, bins = build_hist_from_sample(area, UNIFIED_SAMPLE_SIZE)
            else:
                hist, bins = build_full_hist(area)
            target_histograms[key] = hist

    if not target_histograms:
        print("❌ 未找到任何可用的目标直方图，请检查 DATASET1 路径与数据。")
        return
    print(f"✅ DATASET1 可对比曲线数：{len(target_histograms)}")

    # 读取稳定性映射（可选）
    stability_map, global_75th = load_stability_map(STABILITY_XLSX) if USE_STABILITY else ({}, None)
    if USE_STABILITY:
        print(f"🔗 使用稳定性策略：{STABILITY_STRATEGY}")
        if global_75th is not None:
            print(f"   - 全局75分位推荐：{global_75th}")

    # 针对每个参考组、每个抽样量（或稳定性策略），进行一次内部匹配
    for day in ALL_GROUPS_DAYS:
        for data_folder in ALL_GROUPS_DATAFOLDERS:
            ref_key = f"{day}_{data_folder}"
            area = load_area_series(DATASET1_BASE, day, data_folder)
            if area is None:
                print(f"⚠️ 跳过 {ref_key}（数据不足或未找到）。")
                continue

            # 决定本次要测试的抽样量列表
            if USE_UNIFIED_SAMPLE_SIZE:
                if area.size < UNIFIED_SAMPLE_SIZE:
                    print(f"⚠️ 跳过 {ref_key}（细胞数 < {UNIFIED_SAMPLE_SIZE}）。")
                    continue
                sample_sizes_to_use = [UNIFIED_SAMPLE_SIZE]
            elif USE_STABILITY:
                if STABILITY_MODE == 'from_excel':
                    # 根据策略生成样本量列表（以便与多档固定量并行输出一致，仍使用列表）
                    if STABILITY_STRATEGY == 'per_file':
                        ss = stability_map.get(ref_key, None)
                        sample_sizes_to_use = [ss] if ss and ss > 0 else []
                        if not sample_sizes_to_use:
                            print(f"   → {ref_key} 无稳定点，回退使用固定SAMPLE_SIZES。")
                            sample_sizes_to_use = SAMPLE_SIZES
                    elif STABILITY_STRATEGY == 'global_75th':
                        sample_sizes_to_use = [global_75th] if global_75th else SAMPLE_SIZES
                    elif STABILITY_STRATEGY == 'global_90th':
                        sample_sizes_to_use = [int(np.percentile(list(stability_map.values()), 90))] if stability_map else SAMPLE_SIZES
                    elif STABILITY_STRATEGY == 'global_max':
                        sample_sizes_to_use = [max(stability_map.values())] if stability_map else SAMPLE_SIZES
                    else:
                        sample_sizes_to_use = SAMPLE_SIZES
                else:  # vs_full 模式：与全量曲线比较估计稳定样本量
                    ss = estimate_stable_size_vs_full(area)
                    if ss is None:
                        print(f"   → {ref_key} 未估计出稳定样本量，回退固定SAMPLE_SIZES。")
                        sample_sizes_to_use = SAMPLE_SIZES
                    else:
                        sample_sizes_to_use = [ss]
                        print(f"   → {ref_key} vs_full 稳定样本量估计：{ss}")
            else:
                sample_sizes_to_use = SAMPLE_SIZES

            for sample_size in sample_sizes_to_use:
                ref_hist, bin_centers = build_hist_from_sample(area, sample_size)
                if ref_hist is None:
                    print(f"⚠️ {ref_key} 少于 {sample_size} 个细胞，跳过该抽样量。")
                    continue

                # 计算所有目标与当前参考的多指标相似度
                records = []
                for tgt_key, tgt_hist in target_histograms.items():
                    row = {"Compared Folder": tgt_key}
                    for method in SIMILARITY_METHODS:
                        val = compute_similarity(ref_hist, tgt_hist, method, bin_centers)
                        row[DISPLAY_NAMES[method]] = val
                    records.append(row)

                result_df = pd.DataFrame(records)

                # 保存排序结果（Excel -> CSV 回退）
                base_name = f"intra_ds1_{ref_key}_N{sample_size}".replace(':', '_')
                excel_path = os.path.join(out_dir, f"{base_name}.xlsx")
                try:
                    with pd.ExcelWriter(excel_path) as writer:
                        for method in SIMILARITY_METHODS:
                            col = DISPLAY_NAMES[method]
                            ascending = False if method in HIGHER_BETTER else True
                            result_df.sort_values(by=col, ascending=ascending).to_excel(writer, sheet_name=method, index=False)
                    excel_saved = True
                except OSError as e:
                    excel_saved = False
                    print(f"⚠️ 写入 Excel 失败：{e}。导出为 CSV。")
                    for method in SIMILARITY_METHODS:
                        col = DISPLAY_NAMES[method]
                        ascending = False if method in HIGHER_BETTER else True
                        csv_path = os.path.join(out_dir, f"{base_name}_{method}.csv")
                        result_df.sort_values(by=col, ascending=ascending).to_csv(csv_path, index=False)

                # 控制台输出 TOP_K（按 PLOT_METHOD）
                col_plot = DISPLAY_NAMES[PLOT_METHOD]
                ascending_plot = False if PLOT_METHOD in HIGHER_BETTER else True
                print(f"\n📌 参考 {ref_key} | N={sample_size} | 排序依据：{col_plot}")
                tmp = result_df.sort_values(by=col_plot, ascending=ascending_plot).head(TOP_K)
                for i, (_, row) in enumerate(tmp.iterrows(), start=1):
                    print(f"{i}. {row['Compared Folder']}  |  {col_plot}={row[col_plot]:.4f}")

                # 绘制参考 vs TOP_K 对比曲线
                x_bins = np.linspace(RANGE[0], RANGE[1], BINS + 1)
                x = 0.5 * (x_bins[:-1] + x_bins[1:])
                plt.figure(figsize=(14, 7))
                ref_smooth = gaussian_filter1d(ref_hist, sigma=2)
                plt.plot(x, ref_smooth, label=f"Ref {ref_key} (N={sample_size})", color="black", linewidth=2, zorder=10)

                colors = plt.cm.tab20(np.linspace(0, 1, TOP_K))
                for color, (_, row) in zip(colors, tmp.iterrows()):
                    tgt_hist = target_histograms[row['Compared Folder']]
                    tgt_smooth = gaussian_filter1d(tgt_hist, sigma=2)
                    plt.plot(x, tgt_smooth, label=f"{row['Compared Folder']} ({row[col_plot]:.3f})", color=color, alpha=0.9)

                plt.title(f"DATASET1 Intra Similarity ({col_plot}) | Ref={ref_key} N={sample_size}")
                plt.xlabel("Cell Area")
                plt.ylabel("Normalized Frequency")
                plt.legend(fontsize=8, ncol=2)
                plt.tight_layout()
                fig_path = os.path.join(out_dir, f"{base_name}_top_matches.png")
                try:
                    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
                    plt.close()
                except OSError as e:
                    print(f"⚠️ 保存对比图失败：{e}")

                # 生成两类聚类热力图
                try:
                    similarity_metrics = [
                        DISPLAY_NAMES['cosine'],
                        DISPLAY_NAMES['pearson'],
                        DISPLAY_NAMES['intersection'],
                    ]
                    distance_metrics = [
                        DISPLAY_NAMES['chi_square'],
                        DISPLAY_NAMES['kl'],
                        DISPLAY_NAMES['wasserstein'],
                    ]

                    folders = result_df['Compared Folder']
                    sim_part = result_df[similarity_metrics]
                    dist_part = result_df[distance_metrics]

                    sim_scaled = pd.DataFrame(
                        MinMaxScaler().fit_transform(sim_part),
                        columns=sim_part.columns,
                        index=folders,
                    )
                    dist_scaled = pd.DataFrame(
                        MinMaxScaler().fit_transform(dist_part),
                        columns=dist_part.columns,
                        index=folders,
                    )

                    g1 = sns.clustermap(sim_scaled, cmap="Reds", annot=True, fmt=".2f", figsize=(12, 8), metric="euclidean", method="ward")
                    # 右侧标签完整显示与可读性优化
                    ax1 = g1.ax_heatmap
                    ax1.yaxis.set_tick_params(labelright=True, labelleft=False, pad=2)
                    import matplotlib.pyplot as _plt
                    _plt.setp(ax1.get_yticklabels(), rotation=0, ha='left', fontsize=9)
                    g1.fig.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.05)
                    g1.fig.suptitle(f"Similarity Clustering | Ref={ref_key} N={sample_size}")
                    heat1_path = os.path.join(out_dir, f"{base_name}_similarity_clustermap.png")
                    g1.savefig(heat1_path, dpi=300, bbox_inches="tight")
                    plt.close(g1.fig)

                    g2 = sns.clustermap(dist_scaled, cmap="Blues_r", annot=True, fmt=".2f", figsize=(12, 8), metric="euclidean", method="ward")
                    ax2 = g2.ax_heatmap
                    ax2.yaxis.set_tick_params(labelright=True, labelleft=False, pad=2)
                    _plt.setp(ax2.get_yticklabels(), rotation=0, ha='left', fontsize=9)
                    g2.fig.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.05)
                    g2.fig.suptitle(f"Distance Clustering | Ref={ref_key} N={sample_size}")
                    heat2_path = os.path.join(out_dir, f"{base_name}_distance_clustermap.png")
                    g2.savefig(heat2_path, dpi=300, bbox_inches="tight")
                    plt.close(g2.fig)
                except OSError as e:
                    print(f"⚠️ 保存热力图失败：{e}")

                # 记录小结
                if 'excel_path' in locals() and os.path.exists(excel_path):
                    print(f"📁 排序结果：{excel_path}")
                else:
                    print(f"📁 排序结果：{out_dir} 下 {base_name}_<method>.csv")
                if os.path.exists(fig_path):
                    print(f"🖼️ 对比图：{fig_path}")


if __name__ == "__main__":
    main()


