import os
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


# 基本参数（DATASET2）
DATASET2_BASE = r"D:\project\2-spring\DATASET2\data"
AREA_COL = "area"
BINS = 50
RANGE = (500, 3500)
ALL_GROUPS_DAYS = [f"DAY{i}" for i in range(2, 8)]  # DAY2-DAY7
ALL_GROUPS_DATAFOLDERS = [f"data{j}" for j in range(1, 7)]  # data1-data6
SAMPLE_SIZES = [2500, 3000, 3500, 4000, 4500, 5000]
TOP_K = 10
RESULT_DIR = os.path.join("similarity", "results", "intra_dataset2")

# 稳定性接入（可选）
USE_STABILITY = True
STABILITY_XLSX = r"D:\project\2-spring\DATASET2\code\stability\dataset2_stability_analysis.xlsx"
# 'per_file' | 'global_75th' | 'global_90th' | 'global_max'
STABILITY_STRATEGY = 'per_file'

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

# 统一抽样绘制分布图设置（DATASET2 按 4800 绘制，不足者跳过）
USE_UNIFIED_SAMPLE_SIZE = True
UNIFIED_SAMPLE_SIZE = 4800

# 曲线图标签中显示的指标（按顺序，避免过长建议≤4项）
METRICS_IN_LABEL = ['intersection', 'cosine', 'pearson']
SHORT_METRIC_NAMES = {
    'intersection': 'Int',
    'cosine': 'Cos',
    'pearson': 'Pear',
    'chi_square': 'Chi2',
    'kl': 'KL',
    'wasserstein': 'Wass',
}


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def get_writable_output_dir(preferred_dir: str, min_free_mb: int = 50) -> str:
    env_dir = os.environ.get("INTRA_DATASET2_OUTDIR")
    candidates = []
    if env_dir:
        candidates.append(env_dir)
    candidates.append(preferred_dir)
    candidates.append(os.path.join(tempfile.gettempdir(), "intra_dataset2"))

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


def main():
    # 可复现性
    np.random.seed(42)

    out_dir = get_writable_output_dir(RESULT_DIR, min_free_mb=50)
    ensure_dir(out_dir)

    # 构建所有目标曲线（统一抽样或完整直方图）
    target_histograms: dict[str, np.ndarray] = {}
    for day in ALL_GROUPS_DAYS:
        for data_folder in ALL_GROUPS_DATAFOLDERS:
            key = f"{day}_{data_folder}"
            area = load_area_series(DATASET2_BASE, day, data_folder)
            if area is None:
                continue
            if USE_UNIFIED_SAMPLE_SIZE:
                if area.size < UNIFIED_SAMPLE_SIZE:
                    # 不足 4800 的组不参与
                    continue
                hist, _ = build_hist_from_sample(area, UNIFIED_SAMPLE_SIZE)
            else:
                hist, _ = build_full_hist(area)
            target_histograms[key] = hist

    if not target_histograms:
        print("❌ 未找到任何可用的目标直方图，请检查 DATASET2 路径与数据。")
        return
    print(f"✅ DATASET2 可对比曲线数：{len(target_histograms)}")

    stability_map, global_75th = load_stability_map(STABILITY_XLSX) if USE_STABILITY else ({}, None)
    if USE_STABILITY:
        print(f"🔗 使用稳定性策略：{STABILITY_STRATEGY}")
        if global_75th is not None:
            print(f"   - 全局75分位推荐：{global_75th}")

    for day in ALL_GROUPS_DAYS:
        for data_folder in ALL_GROUPS_DATAFOLDERS:
            ref_key = f"{day}_{data_folder}"
            area = load_area_series(DATASET2_BASE, day, data_folder)
            if area is None:
                print(f"⚠️ 跳过 {ref_key}（数据不足或未找到）。")
                continue

            # 决定本组使用的抽样量列表
            if USE_UNIFIED_SAMPLE_SIZE:
                if area.size < UNIFIED_SAMPLE_SIZE:
                    print(f"⚠️ 跳过 {ref_key}（细胞数 < {UNIFIED_SAMPLE_SIZE}）。")
                    continue
                sample_sizes_to_use = [UNIFIED_SAMPLE_SIZE]
            elif USE_STABILITY:
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
            else:
                sample_sizes_to_use = SAMPLE_SIZES

            for sample_size in sample_sizes_to_use:
                ref_hist, bin_centers = build_hist_from_sample(area, sample_size)
                if ref_hist is None:
                    print(f"⚠️ {ref_key} 少于 {sample_size} 个细胞，跳过该抽样量。")
                    continue

                # 计算多指标相似度
                records = []
                for tgt_key, tgt_hist in target_histograms.items():
                    row = {"Compared Folder": tgt_key}
                    for method in SIMILARITY_METHODS:
                        val = compute_similarity(ref_hist, tgt_hist, method, bin_centers)
                        row[DISPLAY_NAMES[method]] = val
                    records.append(row)

                result_df = pd.DataFrame(records)

                # 保存排序（Excel -> CSV回退）
                base_name = f"intra_ds2_{ref_key}_N{sample_size}".replace(':', '_')
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
                    # 组装多指标标签
                    parts = []
                    for m in METRICS_IN_LABEL:
                        col_name = DISPLAY_NAMES.get(m)
                        if col_name in row:
                            parts.append(f"{SHORT_METRIC_NAMES.get(m, m)}={row[col_name]:.3f}")
                    metrics_text = ", ".join(parts) if parts else f"{col_plot}={row[col_plot]:.3f}"
                    plt.plot(x, tgt_smooth, label=f"{row['Compared Folder']} ({metrics_text})", color=color, alpha=0.9)

                plt.title(f"DATASET2 Intra Similarity ({col_plot}) | Ref={ref_key} N={sample_size}")
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

                    g1 = sns.clustermap(
                        sim_scaled,
                        cmap="Reds",
                        annot=True,
                        fmt=".2f",
                        figsize=(12, 8),
                        metric="euclidean",
                        method="ward",
                    )
                    # 右侧标签显示与可读性优化（按 DATASET1 做法）
                    ax1 = g1.ax_heatmap
                    ax1.yaxis.set_tick_params(labelright=True, labelleft=False, pad=2)
                    # 强制显示所有行标签
                    from matplotlib.ticker import FixedLocator
                    ordered = g1.dendrogram_row.reordered_ind if g1.dendrogram_row else list(range(sim_scaled.shape[0]))
                    labels = [str(folders.iloc[i]) for i in ordered]
                    ax1.yaxis.set_major_locator(FixedLocator(np.arange(len(ordered))))
                    ax1.set_yticklabels(labels)
                    plt.setp(ax1.get_yticklabels(), rotation=0, ha='left', fontsize=8)
                    g1.fig.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.05)
                    g1.fig.suptitle(f"Similarity Clustering | Ref={ref_key} N={sample_size}")
                    heat1_path = os.path.join(out_dir, f"{base_name}_similarity_clustermap.png")
                    g1.savefig(heat1_path, dpi=300, bbox_inches="tight")
                    plt.close(g1.fig)

                    g2 = sns.clustermap(
                        dist_scaled,
                        cmap="Blues_r",
                        annot=True,
                        fmt=".2f",
                        figsize=(12, 8),
                        metric="euclidean",
                        method="ward",
                    )
                    ax2 = g2.ax_heatmap
                    ax2.yaxis.set_tick_params(labelright=True, labelleft=False, pad=2)
                    ordered2 = g2.dendrogram_row.reordered_ind if g2.dendrogram_row else list(range(dist_scaled.shape[0]))
                    labels2 = [str(folders.iloc[i]) for i in ordered2]
                    ax2.yaxis.set_major_locator(FixedLocator(np.arange(len(ordered2))))
                    ax2.set_yticklabels(labels2)
                    plt.setp(ax2.get_yticklabels(), rotation=0, ha='left', fontsize=8)
                    g2.fig.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.05)
                    g2.fig.suptitle(f"Distance Clustering | Ref={ref_key} N={sample_size}")
                    heat2_path = os.path.join(out_dir, f"{base_name}_distance_clustermap.png")
                    g2.savefig(heat2_path, dpi=300, bbox_inches="tight")
                    plt.close(g2.fig)
                except OSError as e:
                    print(f"⚠️ 保存热力图失败：{e}")

                # 小结
                if 'excel_path' in locals() and os.path.exists(excel_path):
                    print(f"📁 排序结果：{excel_path}")
                else:
                    print(f"📁 排序结果：{out_dir} 下 {base_name}_<method>.csv")
                if os.path.exists(fig_path):
                    print(f"🖼️ 对比图：{fig_path}")


if __name__ == "__main__":
    main()


