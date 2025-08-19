import os

# 获取当前目录下所有文件夹名
folders = [f for f in os.listdir() if os.path.isdir(f)]

# 将文件夹名称按字母+数字排序（如 '1h', '2h', ..., '72h'）
folders_sorted = sorted(folders, key=lambda x: int(''.join(filter(str.isdigit, x))))

# 格式化输出
formatted = f'folders = {folders_sorted}'
print(formatted)
