import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def plot_smoothed_histograms(day_to_areas, hist_range, num_bins=30, smooth_points=300, title=None, save_dir=None, save_name="areaRatio_combined_smoothed.png"):
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
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_step = 0.05 * (y_range if y_range > 0 else 1.0)  # 更大的纵向步长
        x_step = 0.01 * (x_range if x_range > 0 else 1.0)  # 轻微横向步长

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
            x_try = x_data + dxi * x_step
            y_try = y_data + dyi * y_step
            x_disp, y_disp = ax.transData.transform((x_try, y_try))
            ok = True
            for (px, py) in placed_disp_coords:
                if abs(x_disp - px) < 60 and abs(y_disp - py) < 18:  # 放宽像素容差，减少重叠
                    ok = False
                    break
            if ok:
                t = plt.text(x_try, y_try, text, fontsize=9, color=color, ha='left', va='bottom', clip_on=False)
                placed_disp_coords.append((x_disp, y_disp))
                return t

        # 兜底直接放置
        return plt.text(x_data, y_data, text, fontsize=9, color=color, ha='left', va='bottom', clip_on=False)

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
        place_label_no_overlap(max_x, max_y, f"({max_x:.0f}, {day_label})", color)

        # 控制台输出检查信息
        print(f"{day_label} ratios sum:", float(ratios.sum()))

    plt.xlabel('Cell Area (pixel)')
    plt.ylabel('Cell Area Ratio')
    if title is None:
        title = 'Cell area distribution across days'
    plt.title(title)
    plt.legend()
    plt.xlim(hist_range)
    plt.ylim(bottom=0)

    # x 轴刻度美化（在最大刻度值附加 pixel）
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

    day_files = find_merged_csvs(data_root)
    if len(day_files) == 0:
        print("未发现任何 merged.csv，无法绘图。请先运行合并脚本生成 total/merged.csv。")
        return

    # 读取全部面积，用于自动确定直方图区间
    all_areas_list = []
    day_to_areas = {}
    for day_label, csv_path in day_files:
        areas = read_areas(csv_path)
        day_to_areas[day_label] = areas
        if areas.size > 0:
            all_areas_list.append(areas)

    all_areas = np.concatenate(all_areas_list) if len(all_areas_list) > 0 else np.array([])
    hist_range = compute_auto_hist_range(all_areas, 0.01, 0.99)
    print(f"Auto hist range: {hist_range}")

    title = 'Cell area distribution (auto-range)'
    save_dir = os.path.join(base_dir, 'combined_visualization')
    plot_smoothed_histograms(day_to_areas, hist_range, num_bins=30, smooth_points=300, title=title, save_dir=save_dir)


if __name__ == "__main__":
    main()


