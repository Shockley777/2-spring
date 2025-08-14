import os
import glob

def check_missing_merged_files():
    """检查哪些images文件夹缺少对应的merged.csv文件"""
    # 获取数据目录路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', '..', '..', 'data')
    
    print(f"检查目录: {os.path.abspath(data_dir)}")
    print("=" * 80)
    
    images_folders = []
    merged_files = []
    
    # 找到所有images文件夹
    for root, dirs, files in os.walk(data_dir):
        if os.path.basename(root) == 'images':
            rel_path = os.path.relpath(root, data_dir)
            path_parts = rel_path.split(os.sep)
            
            # 获取完整路径信息
            full_path_key = '/'.join(path_parts[:-1])  # 去掉最后的'images'
            
            images_folders.append({
                'full_path_key': full_path_key,
                'images_path': root,
                'base_path': os.path.dirname(root)  # 去掉images目录，得到基础路径
            })
    
    # 找到所有merged.csv文件
    pattern = os.path.join(data_dir, "**", "total", "merged.csv")
    merged_paths = glob.glob(pattern, recursive=True)
    
    for merged_path in merged_paths:
        rel_path = os.path.relpath(os.path.dirname(os.path.dirname(merged_path)), data_dir)
        
        merged_files.append({
            'full_path_key': rel_path.replace('\\', '/'),  # 统一使用/分隔符
            'merged_path': merged_path
        })
    
    print(f"📁 找到 {len(images_folders)} 个images文件夹")
    print(f"📄 找到 {len(merged_files)} 个merged.csv文件")
    
    # 按路径排序显示
    images_folders.sort(key=lambda x: x['full_path_key'])
    merged_files.sort(key=lambda x: x['full_path_key'])
    
    print(f"\n" + "=" * 80)
    print("📁 所有images文件夹:")
    for i, img in enumerate(images_folders, 1):
        print(f"  {i:2d}. {img['full_path_key']}")
    
    print(f"\n" + "=" * 80)
    print("📄 所有merged.csv文件:")
    for i, merged in enumerate(merged_files, 1):
        print(f"  {i:2d}. {merged['full_path_key']}")
    
    # 检查哪些images文件夹没有对应的merged.csv
    missing_merged = []
    images_set = {img['full_path_key'] for img in images_folders}
    merged_set = {merged['full_path_key'] for merged in merged_files}
    
    missing_merged_paths = images_set - merged_set
    missing_images_paths = merged_set - images_set
    
    print(f"\n" + "=" * 80)
    print("🔍 对比结果:")
    
    if missing_merged_paths:
        print(f"\n❌ 有images但缺少merged.csv的文件夹 ({len(missing_merged_paths)}个):")
        for path in sorted(missing_merged_paths):
            print(f"   {path}")
            # 检查是否有total目录
            base_path = os.path.join(data_dir, path.replace('/', os.sep))
            total_dir = os.path.join(base_path, 'total')
            if os.path.exists(total_dir):
                print(f"      └─ total目录存在")
                files_in_total = os.listdir(total_dir)
                if files_in_total:
                    print(f"      └─ total目录中的文件: {files_in_total}")
                else:
                    print(f"      └─ total目录为空")
            else:
                print(f"      └─ 缺少total目录")
    else:
        print("\n✅ 所有images文件夹都有对应的merged.csv文件")
    
    if missing_images_paths:
        print(f"\n⚠️  有merged.csv但没有images文件夹 ({len(missing_images_paths)}个):")
        for path in sorted(missing_images_paths):
            print(f"   {path}")
    else:
        print("\n✅ 所有merged.csv文件都有对应的images文件夹")
    
    # 详细统计
    print(f"\n" + "=" * 80)
    print("📊 统计汇总:")
    print(f"   - Images文件夹总数: {len(images_folders)}")
    print(f"   - Merged.csv文件总数: {len(merged_files)}")
    print(f"   - 差异数量: {len(missing_merged_paths)}")
    
    return missing_merged_paths, missing_images_paths

if __name__ == "__main__":
    print("🔍 开始检查images文件夹与merged.csv文件的对应关系...")
    missing_merged, missing_images = check_missing_merged_files()
    
    print("\n" + "=" * 80)
    if missing_merged or missing_images:
        print("⚠️  发现不匹配的情况")
        if missing_merged:
            print(f"   - {len(missing_merged)} 个文件夹缺少merged.csv")
        if missing_images:
            print(f"   - {len(missing_images)} 个merged.csv缺少images文件夹")
    else:
        print("✅ 所有文件夹都匹配完好！") 