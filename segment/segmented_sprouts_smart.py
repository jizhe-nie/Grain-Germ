import cv2
import numpy as np
import os
import glob
from sklearn.cluster import KMeans
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_ROOT = "../output/segmented_seeds_yolo_rembg"
OUTPUT_ROOT = "../output/segmented_sprouts_smart"
DEBUG_MODE = True

# 【核心阈值】聚类中心距离阈值
# 如果两个颜色的差异小于此值，认为没有发芽
# 经验值：在 Lab 空间下，20-40 左右比较合适
COLOR_DISTANCE_THRESHOLD = 25.0


# ===========================================

def segment_sprout_smart(image):
    """
    智能分割：Lab空间 + 距离门控 + 形态学去噪
    """
    # 1. 提取有效像素
    if image.shape[2] == 4:
        b, g, r, a = cv2.split(image)
        mask = a > 0
        img_bgr = image[:, :, :3]
    else:
        # 如果是jpg，生成一个简单的mask
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        img_bgr = image

    valid_pixels_bgr = img_bgr[mask > 0]
    if len(valid_pixels_bgr) == 0: return None, None, False

    # 2. 转换到 LAB 空间 (关键步骤)
    # 只需要对有效像素转换，减少计算量
    # reshape 为了 cv2.cvtColor 能处理
    valid_pixels_bgr_reshaped = valid_pixels_bgr.reshape(1, -1, 3)
    valid_pixels_lab = cv2.cvtColor(valid_pixels_bgr_reshaped, cv2.COLOR_BGR2LAB)
    valid_pixels_lab = valid_pixels_lab.reshape(-1, 3)

    # 3. K-Means 聚类
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(valid_pixels_lab)
    centers = kmeans.cluster_centers_  # shape (2, 3) -> [L, a, b]

    # 4. 【核心策略】计算两个中心的欧氏距离
    # center[0] 和 center[1] 的色彩距离
    diff = centers[0] - centers[1]
    color_distance = np.sqrt(np.sum(diff ** 2))

    # 打印距离用于调试 (正式运行时可注释)
    # print(f"Cluster Distance: {color_distance:.2f}")

    # 如果距离太近，说明是同一种东西（未发芽）
    if color_distance < COLOR_DISTANCE_THRESHOLD:
        # 全是种子，没有芽
        h, w = image.shape[:2]
        seed_mask = (mask * 255).astype(np.uint8)
        sprout_mask = np.zeros((h, w), dtype=np.uint8)
        return seed_mask, sprout_mask, False  # False 表示未发芽

    # 5. 判断哪个是芽
    # 在 Lab 空间中：
    # L (通道0): 亮度。芽通常比种子亮。
    # b (通道2): 黄蓝色。种子偏黄(值高)，芽偏白(值低，接近128中性，或偏蓝)。

    # 评分公式：Score = L - b (越亮且越不黄，越可能是芽)
    # 注意 OpenCv Lab 范围: L[0..255], a[0..255], b[0..255]
    score0 = centers[0][0] - centers[0][2]
    score1 = centers[1][0] - centers[1][2]

    if score0 > score1:
        sprout_label = 0
        seed_label = 1
    else:
        sprout_label = 1
        seed_label = 0

    # 6. 重建 Mask
    h, w = image.shape[:2]
    sprout_mask = np.zeros((h, w), dtype=np.uint8)
    seed_mask = np.zeros((h, w), dtype=np.uint8)

    full_labels = np.full((h, w), -1, dtype=int)
    full_labels[mask > 0] = labels

    raw_sprout_mask = np.zeros((h, w), dtype=np.uint8)
    raw_sprout_mask[full_labels == sprout_label] = 255

    raw_seed_mask = np.zeros((h, w), dtype=np.uint8)
    raw_seed_mask[full_labels == seed_label] = 255

    # 7. 【形态学去噪】去除种子上的高光碎斑
    # 芽应该是连续的，而误判的高光通常是细小的
    # 开运算：先腐蚀(去除小点)再膨胀(恢复形状)
    kernel_clean = np.ones((3, 3), np.uint8)
    sprout_mask_clean = cv2.morphologyEx(raw_sprout_mask, cv2.MORPH_OPEN, kernel_clean, iterations=1)

    # 连通域分析：只保留比较大的块
    num_labels, labels_stats, stats, _ = cv2.connectedComponentsWithStats(sprout_mask_clean)
    final_sprout_mask = np.zeros_like(sprout_mask_clean)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # 如果面积太小（比如单纯是一个反光点），丢弃
        if area > 20:
            final_sprout_mask[labels_stats == i] = 255

    # 种子掩膜 = 原始前景 - 最终芽掩膜
    final_seed_mask = cv2.bitwise_and((mask * 255).astype(np.uint8), cv2.bitwise_not(final_sprout_mask))

    return final_seed_mask, final_sprout_mask, True


def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    all_images = glob.glob(os.path.join(INPUT_ROOT, "**", "*.png"), recursive=True)
    if not all_images:
        all_images = glob.glob(os.path.join(INPUT_ROOT, "**", "*.jpg"), recursive=True)

    print(f"开始智能处理，共 {len(all_images)} 张...")

    for img_path in tqdm(all_images):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None: continue

        # 智能分割
        seed_mask, sprout_mask, has_sprouted = segment_sprout_smart(img)

        if seed_mask is None: continue

        rel_path = os.path.relpath(img_path, INPUT_ROOT)
        base_name = os.path.splitext(rel_path)[0]

        # 只有发芽的才保存 sprout 图片，未发芽的只保存 vis 图或者跳过
        if has_sprouted:
            # 保存芽
            if img.shape[2] == 4:
                b, g, r, a = cv2.split(img)
            else:
                b, g, r = cv2.split(img)

            sprout_rgba = cv2.merge([b, g, r, sprout_mask])
            save_p_sprout = os.path.join(OUTPUT_ROOT, f"{base_name}_sprout.png")
            os.makedirs(os.path.dirname(save_p_sprout), exist_ok=True)
            cv2.imwrite(save_p_sprout, sprout_rgba)

        # 可视化 (红=种子，绿=芽)
        if DEBUG_MODE:
            vis = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            vis[seed_mask > 0] = [0, 0, 255]  # 红
            vis[sprout_mask > 0] = [0, 255, 0]  # 绿

            # 叠加在原图上
            if img.shape[2] == 4:
                orig_vis = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                orig_vis = img

            # 简单的融合
            debug_view = cv2.addWeighted(orig_vis, 0.7, vis, 0.3, 0)

            # 如果未发芽，在这个图上写个文字标记
            if not has_sprouted:
                cv2.putText(debug_view, "No Sprout", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            combined = np.hstack([orig_vis, debug_view])
            save_p_vis = os.path.join(OUTPUT_ROOT, f"{base_name}_vis.jpg")
            os.makedirs(os.path.dirname(save_p_vis), exist_ok=True)
            cv2.imwrite(save_p_vis, combined)


if __name__ == "__main__":
    main()