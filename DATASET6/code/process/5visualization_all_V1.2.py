import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
try:
    from scipy.interpolate import make_interp_spline
except Exception:
    make_interp_spline = None


def find_merged_csvs(data_root):
    day_dirs = sorted(glob.glob(os.path.join(data_root, "DAY*")))
    results = []
    for day_dir in day_dirs:
        csv_path = os.path.join(day_dir, "total", "merged.csv")
        if os.path.exists(csv_path):
            day_label = os.path.basename(day_dir).lower()
            results.append((day_label, csv_path))
    return results


def build_replicate_map_across_days(data_root, day_start=1, day_end=8):
    """
    构建并行实验条件到各天文件路径的映射：
    返回字典：{ rep_tag: { day_index: csv_path, ... }, ... }
    仅收集 DAY[day_start..day_end] 中形如 'merged <rep>-DAYk.csv' 的文件。
    """
    replicate_to_daypath = {}
    day_dirs = sorted(glob.glob(os.path.join(data_root, "DAY*")))
    for day_dir in day_dirs:
        day_name = os.path.basename(day_dir).lower()
        m = re.match(r"day(\d+)", day_name)
        if not m:
            continue
        day_index = int(m.group(1))
        if day_index < day_start or day_index > day_end:
            continue

        total_dir = os.path.join(day_dir, "total")
        if not os.path.isdir(total_dir):
            continue

        for p in sorted(glob.glob(os.path.join(total_dir, "merged *.csv"))):
            base = os.path.basename(p)
            if base == "merged.csv":
                continue
            mm = re.match(r"merged\s+(.+)-DAY(\d+)\.csv$", base, re.IGNORECASE)
            if not mm:
                continue
            rep_tag = mm.group(1).strip()
            day_num_in_name = int(mm.group(2))
            if day_num_in_name != day_index:
                continue
            if rep_tag not in replicate_to_daypath:
                replicate_to_daypath[rep_tag] = {}
            replicate_to_daypath[rep_tag][day_index] = p

    return replicate_to_daypath

def find_merged_csvs_with_replicates(data_root):
    """
    返回 [(label, csv_path), ...]
    - DAY0: 使用 total/merged.csv，label 例如 'day0'
    - 从 DAY1 起：匹配 total/ 目录下形如 'merged R*-DAYk.csv' 的 5 个并行实验文件
      label 例如 'day1-R50', 'day2-R160' 等
    """
    day_dirs = sorted(glob.glob(os.path.join(data_root, "DAY*")))
    results = []
    for day_dir in day_dirs:
        day_name = os.path.basename(day_dir).lower()  # e.g., 'day0'
        m = re.match(r"day(\d+)", day_name)
        day_index = int(m.group(1)) if m else None

        total_dir = os.path.join(day_dir, "total")
        if not os.path.isdir(total_dir):
            continue

        # day0: 只取 merged.csv
        if day_index == 0:
            csv_path = os.path.join(total_dir, "merged.csv")
            if os.path.exists(csv_path):
                results.append((day_name, csv_path))
            continue

        # day1+：查找并行实验文件，跳过汇总的 merged.csv
        candidate_paths = sorted(glob.glob(os.path.join(total_dir, "merged *.csv")))
        for p in candidate_paths:
            base = os.path.basename(p)
            if base == "merged.csv":
                continue
            # 期望格式：merged <rep>-DAY<k>.csv，例如 merged R160-DAY1.csv
            mm = re.match(r"merged\s+(.+)-DAY(\d+)\.csv$", base, re.IGNORECASE)
            if not mm:
                continue
            rep_tag = mm.group(1).strip()
            day_num_in_name = int(mm.group(2))
            # 校验 day 号一致
            if day_index is not None and day_num_in_name != day_index:
                continue
            label = f"{day_name}-{rep_tag}"
            results.append((label, p))

    return results


def find_day1_replicates_else_merged(data_root):
    """
    返回 [(label, csv_path), ...]
    - DAY0 与 DAY>=2: 使用 total/merged.csv，label 为 'dayk'
    - 仅 DAY1: 使用 total/ 下形如 'merged R*-DAY1.csv' 的并行实验文件，
      label 形如 'day1-R160'
    """
    day_dirs = sorted(glob.glob(os.path.join(data_root, "DAY*")))
    results = []
    for day_dir in day_dirs:
        day_name = os.path.basename(day_dir).lower()
        m = re.match(r"day(\d+)", day_name)
        day_index = int(m.group(1)) if m else None

        total_dir = os.path.join(day_dir, "total")
        if not os.path.isdir(total_dir):
            continue

        # 仅 DAY1 使用并行 R 文件
        if day_index == 1:
            candidate_paths = sorted(glob.glob(os.path.join(total_dir, "merged *.csv")))
            has_rep = False
            for p in candidate_paths:
                base = os.path.basename(p)
                if base == "merged.csv":
                    continue
                mm = re.match(r"merged\s+(.+)-DAY(\d+)\.csv$", base, re.IGNORECASE)
                if not mm:
                    continue
                rep_tag = mm.group(1).strip()
                day_num_in_name = int(mm.group(2))
                if day_num_in_name != 1:
                    continue
                label = f"{day_name}-{rep_tag}"
                results.append((label, p))
                has_rep = True
            # 若未发现并行文件，退回 merged.csv
            if not has_rep:
                merged_path = os.path.join(total_dir, "merged.csv")
                if os.path.exists(merged_path):
                    results.append((day_name, merged_path))
            continue

        # 其他天：使用 merged.csv 单曲线
        merged_path = os.path.join(total_dir, "merged.csv")
        if os.path.exists(merged_path):
            results.append((day_name, merged_path))

    return results

def read_areas(csv_path):
    df = pd.read_csv(csv_path)
    if "area" not in df.columns:
        return np.array([])
    return df["area"].to_numpy(dtype=float)


def compute_auto_hist_range(all_areas, lower_q=0.01, upper_q=0.99):
    if all_areas.size == 0:
        # 默认回退到一个合理范围
        return (0.0, 1.0)
    lo = float(np.quantile(all_areas, lower_q))
    hi = float(np.quantile(all_areas, upper_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        # 退回到 min/max
        lo = float(np.nanmin(all_areas))
        hi = float(np.nanmax(all_areas))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            return (0.0, 1.0)
    return (lo, hi)


def plot_smoothed_histograms(day_to_areas, hist_range, num_bins=30, smooth_points=300, title=None, save_dir=None, save_name="areaRatio_combined_smoothed.png", auto_adjust_x=True):
    plt.figure(figsize=(10, 6))

    # 为不同天数准备颜色
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(color_cycle) < len(day_to_areas):
        # 扩展颜色数
        import itertools
        color_cycle = list(itertools.islice(itertools.cycle(["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]), len(day_to_areas)))

    # 简单的标注避让：在像素坐标系下检测重叠，优先向上错位，必要时左右/向下微调
    ax = plt.gca()
    ax.margins(y=0.15)
    placed_disp_coords = []  # 存放已放置文本的像素坐标

    def place_label_no_overlap(x_data, y_data, text, color, max_tries=60):
        # 轴范围与步长
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        x_range = x_max - x_min
        y_step = 0.05 * (y_range if y_range > 0 else 1.0)
        x_step = 0.01 * (x_range if x_range > 0 else 1.0)

        # 让文本始终在黑框内：设置边距并在尝试时夹紧坐标
        margin_x = 0.02 * (x_range if x_range > 0 else 1.0)
        margin_y = 0.05 * (y_range if y_range > 0 else 1.0)

        # 候选偏移序列（优先向上，然后上+右、上+左，最后少量向下）
        candidates = [(0, 0)]
        for k in range(1, 8):
            candidates.append((0, k))
            candidates.append((+1, k))
            candidates.append((-1, k))
        for k in range(1, 4):
            candidates.append((0, -k))
            candidates.append((+1, -k))
            candidates.append((-1, -k))

        for (dxi, dyi) in candidates[:max_tries]:
            # 偏移尝试
            x_try = x_data + dxi * x_step
            y_try = y_data + dyi * y_step
            # 边界夹紧，确保文本锚点留在黑框内
            x_try = min(max(x_try, x_min + margin_x), x_max - margin_x)
            y_try = min(max(y_try, y_min + margin_y), y_max - margin_y)

            x_disp, y_disp = ax.transData.transform((x_try, y_try))
            ok = True
            for (px, py) in placed_disp_coords:
                if abs(x_disp - px) < 60 and abs(y_disp - py) < 18:
                    ok = False
                    break
            if ok:
                # 居中对齐，避免靠近右边界时溢出
                t = plt.text(x_try, y_try, text, fontsize=9, color=color, ha='center', va='bottom', clip_on=True)
                placed_disp_coords.append((x_disp, y_disp))
                return t

        # 兜底：使用夹紧后的原位置
        x_fallback = min(max(x_data, x_min + margin_x), x_max - margin_x)
        y_fallback = min(max(y_data, y_min + margin_y), y_max - margin_y)
        return plt.text(x_fallback, y_fallback, text, fontsize=9, color=color, ha='center', va='bottom', clip_on=True)

    for (idx, (day_label, areas)) in enumerate(day_to_areas.items()):
        if areas.size == 0:
            continue

        counts, bin_edges = np.histogram(areas, bins=num_bins, range=hist_range)
        if counts.sum() == 0:
            # 无有效数据
            continue
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ratios = counts / np.sum(counts)

        try:
            if make_interp_spline is not None and bin_centers.size >= 4:
                spline = make_interp_spline(bin_centers, ratios, k=3)
                x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), smooth_points)
                y_smooth = spline(x_smooth)
            else:
                raise RuntimeError("spline_unavailable")
        except Exception:
            x_smooth = bin_centers
            y_smooth = ratios

        color = color_cycle[idx % len(color_cycle)]
        plt.plot(x_smooth, y_smooth, linestyle='-', color=color, label=day_label)

        # 标注峰值位置（自动避让 + 简写）
        max_idx = int(np.argmax(y_smooth))
        max_x = float(x_smooth[max_idx])
        max_y = float(y_smooth[max_idx])
        place_label_no_overlap(max_x, max_y, f"{int(max_x)}", color)

        # 控制台输出检查信息
        print(f"{day_label} ratios sum:", float(ratios.sum()))

    plt.xlabel('Cell Area (pixel)')
    plt.ylabel('Cell Area Ratio')
    if title is None:
        title = 'Cell area distribution across days'
    plt.title(title)
    plt.legend()
    if not auto_adjust_x:
        plt.xlim(hist_range)
    plt.ylim(bottom=0)

    # 横坐标自动调整：不强制重写刻度，启用适度边距与智能刻度数
    if auto_adjust_x:
        ax = plt.gca()
        ax.margins(x=0.02)
        ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', prune=None))
    else:
        # 保持原有的刻度美化（仅当不启用自动调整时）
        xticks = plt.xticks()[0]
        new_labels = []
        if len(xticks) > 0:
            xmax = xticks.max()
        else:
            xmax = hist_range[1]
        for x in xticks:
            if np.isclose(x, xmax, atol=1e-6):
                new_labels.append(f"{int(x)} pixel")
            else:
                new_labels.append(str(int(x)))
        plt.xticks(xticks, new_labels)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    plt.show()


def main():
    # 从 process/ 跳到 数据集根目录
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_root = os.path.join(base_dir, "data")
    if not os.path.isdir(data_root):
        raise SystemExit(f"未找到数据目录: {data_root}")

    # 每个实验条件单独出图：Rxx day1-8 + day0 baseline
    save_dir = os.path.join(base_dir, 'combined_visualization')

    # day0 baseline
    day0_csv = os.path.join(data_root, "DAY0", "total", "merged.csv")
    if not os.path.exists(day0_csv):
        print(f"未发现 DAY0 baseline: {day0_csv}")
        return
    day0_areas = read_areas(day0_csv)

    replicate_map = build_replicate_map_across_days(data_root, day_start=1, day_end=8)
    if len(replicate_map) == 0:
        print("未发现任何并行实验的 merged R*-DAYk.csv 文件。")
        return

    # 统一横坐标范围 0-35000
    hist_range = (0.0, 35000.0)
    print(f"Using fixed x-range for all figures: {hist_range}")

    for rep_tag in sorted(replicate_map.keys()):
        # 组装该实验条件的 day0..day8 面积数据
        day_to_areas = {"day0": day0_areas}
        for k in range(1, 9):
            p = replicate_map.get(rep_tag, {}).get(k)
            if not p:
                continue
            areas = read_areas(p)
            day_to_areas[f"day{k}"] = areas

        if all(v.size == 0 for v in day_to_areas.values()):
            continue

        title = f"Cell area distribution - {rep_tag}"
        safe_tag = rep_tag.replace(" ", "_").replace("/", "-")
        save_name = f"areaRatio_{safe_tag}_day0-8.png"

        plot_smoothed_histograms(
            day_to_areas,
            hist_range,
            num_bins=20,
            smooth_points=300,
            title=title,
            save_dir=save_dir,
            save_name=save_name,
            auto_adjust_x=False,
        )


if __name__ == "__main__":
    main()


