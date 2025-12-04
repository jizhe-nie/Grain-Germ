import os
import json
import shutil
import random
import glob
from pathlib import Path

# ================= 配置区域 =================
# 1. JSON 标注文件所在的文件夹名称
JSON_DIR = "../dataset/label_seed_tip"

# 2. 原始图片所在的文件夹名称
IMAGE_DIR = "../dataset/germination"

# 3. 输出的 YOLO 数据集路径
OUTPUT_DIR = "../dataset/label_seed_tip_yolo"

# 4. 目标标签 (我们只取 seed 框用于阶段一训练)
TARGET_LABEL = "seed"

# 5. 训练集比例
TRAIN_RATIO = 0.8


# ===========================================

def convert(json_dir, image_dir, output_dir, target_label):
    # 1. 创建 YOLO 目录结构
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # 2. 获取所有 json 文件
    # 使用 glob 获取 json_dir 下所有的 .json 文件
    json_files = glob.glob(os.path.join(json_dir, "*.json"))

    if not json_files:
        print(f"[错误] 在 {json_dir} 下没找到任何 JSON 文件，请检查路径。")
        return

    print(f"找到 {len(json_files)} 个 JSON 文件。准备开始转换...")

    # 打乱顺序
    random.shuffle(json_files)

    count_success = 0
    count_fail = 0

    for i, json_path in enumerate(json_files):
        # 决定划分
        split = 'train' if i < len(json_files) * TRAIN_RATIO else 'val'

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[错误] 无法读取 {json_path}: {e}")
            continue

        # --- 路径匹配逻辑修改 ---
        # LabelMe 的 imagePath 可能是绝对路径，也可能是相对路径
        # 我们只取文件名，然后去 IMAGE_DIR 里找
        raw_img_name = data['imagePath']
        # 强制替换反斜杠，解决 Windows 路径在 Linux 下无法识别的问题
        img_filename = os.path.basename(raw_img_name.replace('\\', '/'))

        # 拼接真正的图片路径
        src_img_path = os.path.join(image_dir, img_filename)

        # 容错检查：如果图片扩展名大小写不一致 (比如 json里是 .JPG, 文件夹里是 .jpg)
        if not os.path.exists(src_img_path):
            # 尝试把后缀换成小写再找一次
            base, ext = os.path.splitext(img_filename)
            src_img_path_lower = os.path.join(image_dir, base + ext.lower())
            if os.path.exists(src_img_path_lower):
                src_img_path = src_img_path_lower
            else:
                # 再尝试换成大写
                src_img_path_upper = os.path.join(image_dir, base + ext.upper())
                if os.path.exists(src_img_path_upper):
                    src_img_path = src_img_path_upper

        if not os.path.exists(src_img_path):
            print(f"[警告] 图片缺失: 在 {image_dir} 中没找到 {img_filename} (对应 {os.path.basename(json_path)})")
            count_fail += 1
            continue

        # 获取图像尺寸
        img_w = data['imageWidth']
        img_h = data['imageHeight']

        # 准备转换标签
        yolo_lines = []
        has_seed = False

        for shape in data['shapes']:
            label = shape['label']

            # 依然只处理 seed 矩形框
            if label == target_label and shape['shape_type'] == 'rectangle':
                has_seed = True
                points = shape['points']

                x1 = min(points[0][0], points[1][0])
                y1 = min(points[0][1], points[1][1])
                x2 = max(points[0][0], points[1][0])
                y2 = max(points[0][1], points[1][1])

                # 坐标归一化
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1

                norm_cx = center_x / img_w
                norm_cy = center_y / img_h
                norm_w = w / img_w
                norm_h = h / img_h

                # 防止越界 (0-1)
                norm_cx = max(0, min(1, norm_cx))
                norm_cy = max(0, min(1, norm_cy))
                norm_w = max(0, min(1, norm_w))
                norm_h = max(0, min(1, norm_h))

                yolo_lines.append(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")

        if has_seed:
            # 1. 复制图片到 datasets/images/train 或 val
            dst_img_name = os.path.basename(src_img_path)
            dst_img_path = os.path.join(output_dir, 'images', split, dst_img_name)
            shutil.copy(src_img_path, dst_img_path)

            # 2. 写入 TXT 标签到 datasets/labels/train 或 val
            # 标签文件名必须和图片名一致 (除了后缀)
            txt_filename = os.path.splitext(dst_img_name)[0] + ".txt"
            dst_txt_path = os.path.join(output_dir, 'labels', split, txt_filename)

            with open(dst_txt_path, 'w') as out_f:
                out_f.write("\n".join(yolo_lines))

            count_success += 1
        else:
            # 只有 tip 没有 seed 的情况
            # print(f"[跳过] {json_path} 不包含 '{target_label}' 标签")
            pass

    print(f"\n处理结束 summary:")
    print(f"成功转换: {count_success} 张")
    print(f"缺失图片: {count_fail} 张")
    print(f"数据集已保存至: {output_dir}")


if __name__ == "__main__":
    convert(JSON_DIR, IMAGE_DIR, OUTPUT_DIR, TARGET_LABEL)