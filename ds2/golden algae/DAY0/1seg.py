import os
import cv2
import numpy as np
from cellpose import models, io

def imread_unicode(file_path):
    """
    使用 np.fromfile 和 cv2.imdecode 来读取含有中文路径的图像文件
    """
    try:
        # 从文件读取原始字节数据
        data = np.fromfile(file_path, dtype=np.uint8)
        # 利用 cv2.imdecode 解码
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"读取图像时出错: {e}")
        return None

def segment_and_save_masks(input_folder, output_folder, model_type='cyto', use_gpu=True):
    """
    遍历 input_folder 中所有图像，使用 cellpose 模型进行细胞分割，
    并将生成的 mask 数据保存到 output_folder 中。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载 cellpose 模型（支持 GPU 加速）
    model = models.Cellpose(model_type=model_type, gpu=use_gpu)

    # 定义有效的图像扩展名
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(input_folder, filename)
            print(f"Processing {img_path}...")

            # 使用自定义的 imread_unicode 读取图像
            img = imread_unicode(img_path)

            if img is None:
                print(f"警告: 无法读取文件 {img_path}，请检查文件路径或文件完整性。")
                continue

            try:
                # 执行分割。channels 参数 [0,0] 表示输入图像为灰度或单通道图像，
                # 如有多通道图像，可修改此参数
                masks, flows, styles, diams = model.eval(img, diameter=48, channels=[0, 0])
            except Exception as e:
                print(f"分割过程中出错: {e}")
                continue

            # 构建 mask 保存路径
            mask_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_mask.png")
            cv2.imwrite(mask_filename, masks.astype(np.uint16))
            print(f"Saved mask to {mask_filename}")

if __name__ == '__main__':
    input_folder = "data1/images"    # 请确保路径正确
    output_folder = "data1/masks"     # 输出 mask 保存的文件夹路径
    segment_and_save_masks(input_folder, output_folder, model_type='cyto3', use_gpu=True)
