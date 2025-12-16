import cv2
import numpy as np
import os
import glob
from sklearn.cluster import KMeans
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_ROOT = "../output/segmented_seeds_yolo_rembg"  # 或者是你刚才那个 perfect_fix 的目录
OUTPUT_ROOT = "../output/segmented_sprouts_final"
DEBUG_MODE = True  # 开启后会保存一张红绿对比图 (红=种子, 绿=芽)


# ===========================================

def segment_sprout_kmeans(image):
    """
    方法：K-Means 聚类
    自动将前景像素分为“深色/黄色组 (种子)”和“浅色/白色组 (芽)”
    """
    # 1. 提取有效像素 (非背景)
    # 假设输入是 BGRA 或 BGR (背景全黑)
    if image.shape[2] == 4:
        b, g, r, a = cv2.split(image)
        mask = a > 0  # 利用透明通道
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)  # 利用亮度
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 只要 mask 为真的地方的像素
    valid_pixels = img_rgb[mask > 0]

    if len(valid_pixels) == 0: return None, None

    # 2. K-Means 聚类 (k=2)
    # 我们希望把像素分为两类：种子(颜色深/饱和度高) 和 芽(颜色浅/白色)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(valid_pixels)

    # 3. 判断哪一类是“芽”
    # 策略：芽通常更亮 (Lightness) 或者 饱和度更低 (Saturation)
    # 这里我们比较两个中心的“亮度”，亮度高的那一类定义为芽
    centers = kmeans.cluster_centers_
    # 计算亮度 (简单均值 R+G+B)
    brightness = np.sum(centers, axis=1)

    sprout_label = np.argmax(brightness)  # 亮度高的是芽
    seed_label = 1 - sprout_label  # 亮度低的是种子

    # 4. 重建 Mask
    h, w = image.shape[:2]
    sprout_mask = np.zeros((h, w), dtype=np.uint8)
    seed_mask = np.zeros((h, w), dtype=np.uint8)

    # 将一维的 labels 映射回二维图片
    # 创建全图的 label 矩阵 (背景设为 -1)
    full_labels = np.full((h, w), -1, dtype=int)
    full_labels[mask > 0] = labels

    sprout_mask[full_labels == sprout_label] = 255
    seed_mask[full_labels == seed_label] = 255

    # 5. 形态学优化 (去除噪点)
    # 种子通常是实心的，芽是细长的
    kernel = np.ones((3, 3), np.uint8)

    # 对种子做闭运算填补空洞
    seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, kernel)
    # 对芽做开运算去除离散噪点
    sprout_mask = cv2.morphologyEx(sprout_mask, cv2.MORPH_OPEN, kernel)

    return seed_mask, sprout_mask


def segment_sprout_hsv(image):
    """
    方法：HSV 阈值 (你的原始想法)
    作为备选方案，手动卡阈值
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 提取前景 mask (V > 10 且 A > 0)
    if image.shape[2] == 4:
        _, _, _, a = cv2.split(image)
        foreground = a > 0
    else:
        _, foreground = cv2.threshold(v, 10, 255, cv2.THRESH_BINARY)
        foreground = foreground > 0

    # 核心逻辑：种子是黄色的 -> 饱和度(S)高
    # 芽是白色的 -> 饱和度(S)低
    # 阈值经验值：40-60 之间 (0-255范围)
    S_THRESHOLD = 50

    seed_mask = np.zeros_like(s)
    sprout_mask = np.zeros_like(s)

    # 种子：是前景 且 S > 阈值
    seed_mask[(foreground) & (s > S_THRESHOLD)] = 255

    # 芽：是前景 且 S <= 阈值
    sprout_mask[(foreground) & (s <= S_THRESHOLD)] = 255

    return seed_mask, sprout_mask


def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    # 读取图片 (支持 png 或 jpg)
    all_images = glob.glob(os.path.join(INPUT_ROOT, "**", "*.png"), recursive=True)
    if not all_images:
        all_images = glob.glob(os.path.join(INPUT_ROOT, "**", "*.jpg"), recursive=True)

    print(f"开始分离芽和种子，共 {len(all_images)} 张...")

    for img_path in tqdm(all_images):
        # 读取带透明通道的图片 (如果是png)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None: continue

        # --- 核心分割 (这里选用 K-Means，更智能) ---
        seed_mask, sprout_mask = segment_sprout_kmeans(img)
        # 如果你想试手动阈值，取消下面这行的注释：
        # seed_mask, sprout_mask = segment_sprout_hsv(img)

        if seed_mask is None: continue

        # --- 保存结果 ---
        # 保持目录结构
        rel_path = os.path.relpath(img_path, INPUT_ROOT)
        base_name = os.path.splitext(rel_path)[0]  # 去掉后缀

        # 1. 保存单纯的芽 (PNG 透明)
        # 只要 sprout_mask 为白的地方，保留原色，其他全透
        if img.shape[2] == 4:
            b, g, r, a = cv2.split(img)
        else:
            b, g, r = cv2.split(img)

        # 芽的图片
        sprout_rgba = cv2.merge([b, g, r, sprout_mask])
        save_p_sprout = os.path.join(OUTPUT_ROOT, f"{base_name}_sprout.png")
        os.makedirs(os.path.dirname(save_p_sprout), exist_ok=True)
        cv2.imwrite(save_p_sprout, sprout_rgba)

        # 2. (可选) 可视化对比图
        if DEBUG_MODE:
            # 种子涂红，芽涂绿，背景黑
            vis = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            vis[seed_mask > 0] = [0, 0, 255]  # 红
            vis[sprout_mask > 0] = [0, 255, 0]  # 绿

            # 原图
            if img.shape[2] == 4:
                orig_vis = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                orig_vis = img

            combined = np.hstack([orig_vis, vis])
            save_p_vis = os.path.join(OUTPUT_ROOT, f"{base_name}_vis.jpg")
            cv2.imwrite(save_p_vis, combined)


if __name__ == "__main__":
    main()