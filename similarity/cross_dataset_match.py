import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wasserstein_distance, pearsonr, entropy
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import tempfile
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# 全局参数
DATASET1_BASE = r"D:\project\2-spring\DATASET1\data"
DATASET2_BASE = r"D:\project\2-spring\DATASET2\data"
AREA_COL = "area"
BINS = 50
RANGE = (500, 3500)
REF_RATIO = 1.0  # 参考库使用全部数据
TARGET_RATIO = 0.8  # 目标曲线抽样比例，可按需调整
TOP_K = 10
RESULT_DIR = os.path.join("similarity", "results", "cross_dataset")

# 可选：手动指定目标曲线键，例如 'DS2::DAY4_data1'；为 None 时随机选择
TARGET_KEY = None

# 可选：绘图与TOPK选择所依据的指标
PLOT_METHOD = 'intersection'  # 可选：intersection/cosine/pearson/chi_square/kl/wasserstein

# 定义相似度/距离方法集合
SIMILARITY_METHODS = [
    'intersection',    # 直方图交集，相似度（越大越好）
    'cosine',          # 余弦相似度（越大越好）
    'pearson',         # 皮尔逊相关（越大越好）
    'chi_square',      # 卡方距离（越小越好）
    'kl',              # KL散度（越小越好）
    'wasserstein',     # Wasserstein距离（越小越好）
]

# 用于打印与保存时的人类可读名称
DISPLAY_NAMES = {
    'intersection': 'Histogram Intersection',
    'cosine': 'Cosine Similarity',
    'pearson': 'Pearson Correlation',
    'chi_square': 'Chi-Square Distance',
    'kl': 'KL Divergence',
    'wasserstein': 'Wasserstein Distance',
}

# 哪些指标越大越好
HIGHER_BETTER = {'intersection', 'cosine', 'pearson'}


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


def compute_similarity(hist1: np.ndarray, hist2: np.ndarray, method: str, bin_centers: np.ndarray) -> float:
    """统一的相似度/距离计算入口。

    说明：除 'intersection' 外，其余方法均基于归一化直方图；
    距离型指标（chi_square/kl/wasserstein）越小越相似。
    """
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
        return float(0.5 * np.sum(((h1 - h2) ** 2) / (denom)))
    elif method == 'kl':
        return float(entropy(h1, h2))
    elif method == 'wasserstein':
        return float(wasserstein_distance(bin_centers, bin_centers, h1, h2))
    else:
        raise ValueError(f"Unsupported similarity method: {method}")


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
    # 仅固定 numpy 的随机性以保证直方图抽样可复现
    np.random.seed(42)
    # 选择输出目录：优先使用环境变量/既定目录，空间不足则自动回退到临时目录
    out_dir = get_writable_output_dir(RESULT_DIR, min_free_mb=50)
    ensure_dir(out_dir)

    # 1) 构建 DATASET1 参考库
    reference_histograms = build_reference_library()
    if not reference_histograms:
        print("❌ 未能从 DATASET1 构建任何参考直方图，请检查数据路径与文件。")
        return
    print(f"✅ 参考库载入成功：{len(reference_histograms)} 条曲线")

    # 2) 收集 DATASET2 可作为目标的曲线列表，并随机选择一条
    target_histograms = gather_dataset2_targets()
    if not target_histograms:
        print("❌ 未能从 DATASET2 载入任何目标直方图，请检查数据路径与文件。")
        return
    if TARGET_KEY and TARGET_KEY in target_histograms:
        target_key = TARGET_KEY
    else:
        # 使用系统熵源，避免受全局随机种子影响
        target_key = random.SystemRandom().choice(list(target_histograms.keys()))
    target_hist = target_histograms[target_key]
    print(f"🎯 随机选取目标曲线：{target_key}")

    # 3) 计算与参考库所有曲线的多指标相似度/距离
    x_bins = np.linspace(RANGE[0], RANGE[1], BINS + 1)
    x = 0.5 * (x_bins[:-1] + x_bins[1:])
    records = []
    for ref_key, ref_hist in reference_histograms.items():
        row = {"Reference": ref_key}
        for method in SIMILARITY_METHODS:
            val = compute_similarity(target_hist, ref_hist, method, x)
            row[DISPLAY_NAMES[method]] = val
        records.append(row)

    result_df = pd.DataFrame(records)

    # 4) 保存到 Excel（每个指标一个 sheet，按对应方向排序）；磁盘不足时回退为 CSV
    excel_path = os.path.join(out_dir, f"match_results_{target_key.replace('::','_')}.xlsx")
    try:
        with pd.ExcelWriter(excel_path) as writer:
            for method in SIMILARITY_METHODS:
                col = DISPLAY_NAMES[method]
                ascending = False if method in HIGHER_BETTER else True
                result_df.sort_values(by=col, ascending=ascending).to_excel(writer, sheet_name=method, index=False)
        excel_saved = True
    except OSError as e:
        excel_saved = False
        print(f"⚠️ 写入 Excel 失败：{e}. 将改为导出 CSV 文件。")
        for method in SIMILARITY_METHODS:
            col = DISPLAY_NAMES[method]
            ascending = False if method in HIGHER_BETTER else True
            csv_path = os.path.join(out_dir, f"match_results_{method}_{target_key.replace('::','_')}.csv")
            result_df.sort_values(by=col, ascending=ascending).to_csv(csv_path, index=False)

    # 5) 终端输出每个指标的 TOP_K
    for method in SIMILARITY_METHODS:
        col = DISPLAY_NAMES[method]
        ascending = False if method in HIGHER_BETTER else True
        print(f"\n🔝 {col} TOP{TOP_K}：")
        tmp = result_df.sort_values(by=col, ascending=ascending).head(TOP_K)
        for i, (_, row) in enumerate(tmp.iterrows(), start=1):
            print(f"{i}. {row['Reference']}  |  {col}={row[col]:.4f}")

    # 6) 绘制目标曲线与PLOT_METHOD对应的TOP_K参考曲线对比图
    plt.figure(figsize=(14, 7))
    target_smooth = gaussian_filter1d(target_hist, sigma=2)
    plt.plot(x, target_smooth, label=f"Target {target_key}", color="black", linewidth=2, zorder=10)

    colors = plt.cm.tab20(np.linspace(0, 1, TOP_K))
    col_plot = DISPLAY_NAMES[PLOT_METHOD]
    ascending_plot = False if PLOT_METHOD in HIGHER_BETTER else True
    top_rows = result_df.sort_values(by=col_plot, ascending=ascending_plot).head(TOP_K)
    for color, (_, row) in zip(colors, top_rows.iterrows()):
        ref_hist = reference_histograms[row['Reference']]
        ref_smooth = gaussian_filter1d(ref_hist, sigma=2)
        plt.plot(x, ref_smooth, label=f"{row['Reference']} ({row[col_plot]:.3f})", color=color, alpha=0.9)

    plt.title(f"Cross-Dataset Similarity ({DISPLAY_NAMES[PLOT_METHOD]}): Target vs Top Matches")
    plt.xlabel("Cell Area")
    plt.ylabel("Normalized Frequency")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"match_plot_{target_key.replace('::','_')}.png")
    try:
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        fig_saved = True
    except OSError as e:
        fig_saved = False
        print(f"⚠️ 保存图片失败：{e}。")

    # 7) 生成聚类热力图（相似度指标与距离指标）
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

        folders = result_df['Reference']
        # 相似度类（越大越好）
        sim_part = result_df[similarity_metrics]
        sim_scaler = MinMaxScaler()
        sim_scaled = pd.DataFrame(
            sim_scaler.fit_transform(sim_part),
            columns=sim_part.columns,
            index=folders,
        )
        g1 = sns.clustermap(
            sim_scaled,
            cmap="Reds",
            annot=True,
            fmt=".2f",
            figsize=(10, 7),
            metric="euclidean",
            method="ward",
        )
        g1.fig.suptitle(f"Similarity Clustering to {target_key} (High=Better)")
        heat1_path = os.path.join(out_dir, f"clustering_similarity_metrics_{target_key.replace('::','_')}.png")
        g1.savefig(heat1_path, dpi=300, bbox_inches="tight")
        plt.close(g1.fig)

        # 距离类（越小越好），为了颜色一致性可做反向或直接展示
        dist_part = result_df[distance_metrics]
        dist_scaler = MinMaxScaler()
        dist_scaled = pd.DataFrame(
            dist_scaler.fit_transform(dist_part),
            columns=dist_part.columns,
            index=folders,
        )
        g2 = sns.clustermap(
            dist_scaled,
            cmap="Blues_r",
            annot=True,
            fmt=".2f",
            figsize=(10, 7),
            metric="euclidean",
            method="ward",
        )
        g2.fig.suptitle(f"Distance Clustering to {target_key} (Low=Better)")
        heat2_path = os.path.join(out_dir, f"clustering_distance_metrics_{target_key.replace('::','_')}.png")
        g2.savefig(heat2_path, dpi=300, bbox_inches="tight")
        plt.close(g2.fig)
        heatmaps_saved = True
    except OSError as e:
        heatmaps_saved = False
        print(f"⚠️ 保存热力图失败：{e}。")

    if excel_saved:
        print(f"\n📁 结果已保存：")
        print(f"- {excel_path}")
    else:
        print(f"\n📁 结果已保存（CSV 回退）：")
        print(f"- {out_dir} 下各个 match_results_<method>_{target_key.replace('::','_')}.csv")
    if 'fig_saved' in locals() and fig_saved:
        print(f"- {fig_path}")
    else:
        print("- 曲线对比图未保存（磁盘空间不足或IO错误）")
    if 'heatmaps_saved' in locals() and heatmaps_saved:
        print(f"- {heat1_path}")
        print(f"- {heat2_path}")
    else:
        print("- 热力图未保存（磁盘空间不足或IO错误）")


if __name__ == "__main__":
    main()


