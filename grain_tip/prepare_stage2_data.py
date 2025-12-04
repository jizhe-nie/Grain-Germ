import os
import json
import cv2
import numpy as np
import glob
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 原始路径
JSON_DIR = "../dataset/label_seed_tip"
IMAGE_DIR = "../dataset/germination"

# 2. 输出路径 (第二阶段数据集)
OUTPUT_ROOT = "../dataset/stage2_data"
OUTPUT_IMGS = os.path.join(OUTPUT_ROOT, "images")
os.makedirs(OUTPUT_IMGS, exist_ok=True)

# 3. 关键点模型输入尺寸 (文档建议 112x112)
TARGET_SIZE = 112

# 4. 标签保存文件 (格式: filename x_norm y_norm)
LABEL_FILE = os.path.join(OUTPUT_ROOT, "keypoint_labels.txt")


# ===========================================

def process_dataset():
    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))
    print(f"找到 {len(json_files)} 个 JSON 文件，开始处理...")

    valid_samples = 0
    data_lines = []

    for json_path in tqdm(json_files):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue

        # --- 1. 读取图片 ---
        # 兼容路径问题
        raw_img_name = data['imagePath']
        img_filename = os.path.basename(raw_img_name.replace('\\', '/'))
        src_img_path = os.path.join(IMAGE_DIR, img_filename)

        # 容错读取
        if not os.path.exists(src_img_path):
            # 尝试大小写
            base, ext = os.path.splitext(img_filename)
            if os.path.exists(os.path.join(IMAGE_DIR, base + ext.lower())):
                src_img_path = os.path.join(IMAGE_DIR, base + ext.lower())
            elif os.path.exists(os.path.join(IMAGE_DIR, base + ext.upper())):
                src_img_path = os.path.join(IMAGE_DIR, base + ext.upper())
            else:
                continue  # 真的找不到，跳过

        img = cv2.imread(src_img_path)
        if img is None:
            continue

        # --- 2. 解析 JSON，配对 Seed 框和 Tip 点 ---
        # 这一步比较关键，因为一个图里可能有多个种子。
        # 我们需要判断哪个 Tip 属于哪个 Seed 框。

        seeds = []  # 存 [x1, y1, x2, y2]
        tips = []  # 存 [tx, ty]

        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']

            if label == 'seed' and shape['shape_type'] == 'rectangle':
                x1 = min(points[0][0], points[1][0])
                y1 = min(points[0][1], points[1][1])
                x2 = max(points[0][0], points[1][0])
                y2 = max(points[0][1], points[1][1])
                seeds.append([x1, y1, x2, y2])

            elif label == 'tip' and shape['shape_type'] == 'point':
                tx, ty = points[0]
                tips.append([tx, ty])

        # --- 3. 匹配逻辑 (判断点是否在框内) ---
        for i, seed_box in enumerate(seeds):
            x1, y1, x2, y2 = seed_box

            # 找到属于这个框的 tip
            matched_tip = None
            for tip in tips:
                tx, ty = tip
                # 简单几何判断：点是否在矩形内 (稍微放宽一点边界以防边缘误差)
                if x1 <= tx <= x2 and y1 <= ty <= y2:
                    matched_tip = tip
                    break

            if matched_tip is None:
                continue  # 这个种子没有标芽尖，跳过

            # --- 4. 裁剪与坐标变换 (核心数学部分) ---
            tx, ty = matched_tip

            # 4.1 裁剪图片
            # 注意：opencv 是 img[y:y+h, x:x+w]
            # 增加一些 padding 防止切到边缘? 暂时先不加，严格按框切
            crop_img = img[int(y1):int(y2), int(x1):int(x2)]

            if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                continue

            # 4.2 坐标平移 (变成相对于小图左上角)
            tx_crop = tx - x1
            ty_crop = ty - y1

            # 4.3 Resize 到 112x112
            h_old, w_old = crop_img.shape[:2]
            crop_resized = cv2.resize(crop_img, (TARGET_SIZE, TARGET_SIZE))

            # 4.4 坐标缩放 (Scale)
            # 新坐标 = 旧坐标 * (新尺寸 / 旧尺寸)
            tx_new = tx_crop * (TARGET_SIZE / w_old)
            ty_new = ty_crop * (TARGET_SIZE / h_old)

            # 4.5 归一化 (0-1) 用于训练
            tx_norm = tx_new / TARGET_SIZE
            ty_norm = ty_new / TARGET_SIZE

            # --- 5. 保存 ---
            # 生成唯一文件名: 原文件名_索引.jpg
            base_name = os.path.splitext(img_filename)[0]
            save_name = f"{base_name}_{i}.jpg"
            save_path = os.path.join(OUTPUT_IMGS, save_name)

            cv2.imwrite(save_path, crop_resized)

            # 记录: 文件名 x_norm y_norm
            data_lines.append(f"{save_name} {tx_norm:.6f} {ty_norm:.6f}")
            valid_samples += 1

    # 保存索引文件
    with open(LABEL_FILE, 'w') as f:
        f.write("\n".join(data_lines))

    print(f"\n处理完成！")
    print(f"共生成 {valid_samples} 个训练样本。")
    print(f"图片保存在: {OUTPUT_IMGS}")
    print(f"标签保存在: {LABEL_FILE}")


if __name__ == "__main__":
    process_dataset()