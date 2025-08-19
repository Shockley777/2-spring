import os
import glob
import argparse

def count_images_in_dataset():
    base_path = "DATASET4"
    total_images = 0
    
    print("=== 计算DATASET4中的图片数量 ===\n")
    
    # 计算20250510文件夹中的图片
    print("1. 计算20250510文件夹:")
    path_20250510 = os.path.join(base_path, "20250510")
    if os.path.exists(path_20250510):
        count_20250510 = count_images_in_folder(path_20250510)
        total_images += count_20250510
        print(f"   20250510总计: {count_20250510} 张图片")
    else:
        print("   20250510文件夹不存在")
    
    print("\n2. 计算TIMECOURSE文件夹:")
    path_timecourse = os.path.join(base_path, "TIMECOURSE")
    if os.path.exists(path_timecourse):
        count_timecourse = count_images_in_folder(path_timecourse)
        total_images += count_timecourse
        print(f"   TIMECOURSE总计: {count_timecourse} 张图片")
    else:
        print("   TIMECOURSE文件夹不存在")
    
    print(f"\n=== 总结 ===")
    print(f"DATASET4总计: {total_images} 张图片")
    
    return total_images

def count_images_in_folder(folder_path):
    """递归计算文件夹中所有.jpg图片的数量"""
    total_count = 0
    
    # 遍历所有子文件夹
    for root, dirs, files in os.walk(folder_path):
        # 检查当前文件夹是否包含images子文件夹
        images_path = os.path.join(root, "images")
        if os.path.exists(images_path):
            # 计算images文件夹中的.jpg文件数量
            image_files = glob.glob(os.path.join(images_path, "*.jpg"))
            count = len(image_files)
            if count > 0:
                # 显示相对路径
                rel_path = os.path.relpath(root, folder_path)
                print(f"   {rel_path}/images: {count} 张图片")
                total_count += count
    
    return total_count

def count_images_generic(base_path: str, dataset_name: str):
    """通用统计：递归搜索 base_path 下所有含 images 子目录的 *.jpg 数量"""
    print(f"=== 计算{dataset_name}中的图片数量 ===\n")
    if not os.path.exists(base_path):
        print(f"路径不存在: {base_path}")
        return 0
    total = count_images_in_folder(base_path)
    print(f"\n=== 总结 ===")
    print(f"{dataset_name}总计: {total} 张图片")
    return total

def count_images_in_folder_ext(folder_path: str, extensions=None, only_in_images_subdir: bool = False, verbose: bool = True):
    """递归统计：支持自定义后缀；可选仅统计名为 images 的子目录。
    extensions: 如 ['.png','.jpg']，默认 ['.jpg']
    """
    if extensions is None:
        extensions = ['.jpg']
    extensions = {e.lower() for e in extensions}

    total_count = 0
    for root, dirs, files in os.walk(folder_path):
        target_dir = root
        if only_in_images_subdir and os.path.basename(root).lower() != 'images':
            continue
        # 统计当前目录下符合扩展名的文件
        count = 0
        for ext in extensions:
            count += len(glob.glob(os.path.join(target_dir, f"*{ext}")))
        if count > 0:
            if verbose:
                rel_path = os.path.relpath(target_dir, folder_path)
                display = rel_path if rel_path != '.' else os.path.basename(folder_path)
                print(f"   {display}: {count} 张图片")
            total_count += count
    return total_count

def count_images_generic_custom(base_path: str, dataset_name: str, exts, only_in_images_subdir: bool):
    print(f"=== 计算{dataset_name}中的图片数量 ===\n")
    if not os.path.exists(base_path):
        print(f"路径不存在: {base_path}")
        return 0
    total = count_images_in_folder_ext(base_path, exts, only_in_images_subdir)
    print(f"\n=== 总结 ===")
    print(f"{dataset_name}总计: {total} 张图片")
    return total

def count_images_group_by_day(base_path: str, dataset_name: str, exts):
    """按 Day*/DAY* 目录汇总统计，不展开到更细层级。"""
    print(f"=== 计算{dataset_name}中的图片数量（按DAY汇总） ===\n")
    if not os.path.exists(base_path):
        print(f"路径不存在: {base_path}")
        return 0
    total = 0
    entries = [d for d in sorted(os.listdir(base_path))
               if os.path.isdir(os.path.join(base_path, d)) and d.strip().lower().startswith('day')]
    if not entries:
        # 若没有 Day* 结构，则直接整体统计
        return count_images_generic_custom(base_path, dataset_name, exts, False)
    for day in entries:
        day_path = os.path.join(base_path, day)
        cnt = count_images_in_folder_ext(day_path, exts, only_in_images_subdir=False, verbose=False)
        print(f"   {day}: {cnt} 张图片")
        total += cnt
    print(f"\n=== 总结 ===")
    print(f"{dataset_name}总计: {total} 张图片")
    return total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计 DATASET4/5/6/7 中 images/*.jpg 数量")
    parser.add_argument(
        "--datasets",
        type=str,
        default="4",
        help="要统计的数据集，逗号分隔（4,5,6,7 或 all）。默认 4",
    )
    args = parser.parse_args()

    targets = [s.strip() for s in ("4,5,6,7" if args.datasets.lower()=="all" else args.datasets).split(",") if s.strip()]

    for t in targets:
        if t == "1":
            count_images_group_by_day(os.path.join("DATASET1", "data"), "DATASET1", ['.png', '.jpg', '.jpeg', '.tif', '.tiff'])
            print("\n")
            continue
        if t == "2":
            count_images_group_by_day(os.path.join("DATASET2", "data"), "DATASET2", ['.png', '.jpg', '.jpeg', '.tif', '.tiff'])
            print("\n")
            continue
        if t == "4":
            count_images_in_dataset()
            print("\n")
        elif t == "5":
            count_images_generic(os.path.join("DATASET5", "data"), "DATASET5")
            print("\n")
        elif t == "6":
            count_images_generic(os.path.join("DATASET6", "data"), "DATASET6")
            print("\n")
        elif t == "7":
            count_images_generic(os.path.join("DATASET7", "data"), "DATASET7")
            print("\n")
        else:
            print(f"未知数据集标识: {t}")