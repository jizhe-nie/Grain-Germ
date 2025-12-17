import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_ROOT = "../output/segmented_seeds_yolo_rembg"  # 上一步的结果目录
OUTPUT_ROOT = "../output/segmented_sprouts_rulebased"
DEBUG_MODE = True

# 【核心参数：黄度阈值】(Lab空间中 b通道, 0-255, 128为中性灰)
# 大于此值 = 偏黄 (种子)
# 小于此值 = 偏白/蓝 (芽)
# 经验值：135-140 之间。设得越高，越容易把浅黄种子误判为芽；设得越低，越容易漏掉芽。
# 针对水稻，138 是一个比较稳的界限。
YELLOW_THRESHOLD = 138

# 【核心参数：亮度阈值】(L通道, 0-255)
# 芽通常比较亮。如果像素太暗，即使不黄，也不是芽（可能是种子的阴影）。
BRIGHTNESS_THRESHOLD = 90


# ===========================================

def segment_sprout_lab_rule(image):
    """
    基于 Lab 空间的物理规则分割
    规则：芽 = (亮度足够高) AND (颜色不够黄)
    """
    h, w = image.shape[:2]

    # 1. 提取 Alpha 通道生成 Mask (排除背景)
    if image.shape[2] == 4:
        b, g, r, a = cv2.split(image)
        # 二值化 alpha 通道，确保掩膜干净
        _, fg_mask = cv2.threshold(a, 10, 255, cv2.THRESH_BINARY)
        img_bgr = image[:, :, :3]
    else:
        # 如果是 jpg，假设背景全黑
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, fg_mask = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY)
        img_bgr = image

    # 如果前景几乎为空，直接返回
    if cv2.countNonZero(fg_mask) == 0:
        return None, None, False

    # 2. 转换到 Lab 空间
    # L: 亮度, a: 红绿, b: 黄蓝
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(img_lab)

    # ========================================================
    # 核心逻辑：定义什么是“芽”
    # 条件1: 不够黄 (B < YELLOW_THRESHOLD) -> 排除颖壳
    # 条件2: 足够亮 (L > BRIGHTNESS_THRESHOLD) -> 排除深色杂质
    # ========================================================

    # 初始芽掩膜
    sprout_condition = (B < YELLOW_THRESHOLD) & (L > BRIGHTNESS_THRESHOLD)

    # 应用前景限制
    raw_sprout_mask = np.zeros_like(fg_mask)
    raw_sprout_mask[sprout_condition & (fg_mask > 0)] = 255

    # 3. 形态学清洗 (去噪)
    # 真正的芽应该是一个连续的块，而不是散落在种子上的孤立噪点
    kernel = np.ones((3, 3), np.uint8)

    # 开运算：先腐蚀后膨胀，去除细小的“高光噪点”
    # iterations=1 比较保守，如果是大图可以设为 2
    sprout_clean = cv2.morphologyEx(raw_sprout_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4. 连通域筛选 (Size Filtering)
    # 如果分割出来的块太小（例如只是种子尖端的一个白点），视为未发芽
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sprout_clean)
    final_sprout_mask = np.zeros_like(sprout_clean)
    has_real_sprout = False

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # 面积阈值：只有大于 50 像素的块才算真芽
        if area > 50:
            final_sprout_mask[labels == i] = 255
            has_real_sprout = True

    # 5. 生成种子掩膜 (前景 - 芽)
    final_seed_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_not(final_sprout_mask))

    return final_seed_mask, final_sprout_mask, has_real_sprout


def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    all_images = glob.glob(os.path.join(INPUT_ROOT, "**", "*.png"), recursive=True)
    if not all_images:
        all_images = glob.glob(os.path.join(INPUT_ROOT, "**", "*.jpg"), recursive=True)

    print(f"开始处理，模式：Lab规则门控 (b < {YELLOW_THRESHOLD})...")

    count_sprouted = 0
    count_total = 0

    for img_path in tqdm(all_images):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None: continue

        count_total += 1

        # 执行分割
        seed_mask, sprout_mask, has_sprout = segment_sprout_lab_rule(img)

        if seed_mask is None: continue

        rel_path = os.path.relpath(img_path, INPUT_ROOT)
        base_name = os.path.splitext(rel_path)[0]

        # 可视化 (调试模式)
        if DEBUG_MODE:
            vis = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            # 红色表示识别为种子，绿色表示识别为芽
            vis[seed_mask > 0] = [0, 0, 255]
            vis[sprout_mask > 0] = [0, 255, 0]

            # 原图
            if img.shape[2] == 4:
                orig = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                orig = img

            # 融合显示
            combined = cv2.addWeighted(orig, 0.6, vis, 0.4, 0)

            # 标记状态
            text = "Sprouted" if has_sprout else "Seed Only"
            color = (0, 255, 0) if has_sprout else (0, 0, 255)
            cv2.putText(combined, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            save_p_vis = os.path.join(OUTPUT_ROOT, f"{base_name}_check.jpg")
            os.makedirs(os.path.dirname(save_p_vis), exist_ok=True)
            cv2.imwrite(save_p_vis, combined)

        # 保存结果 (只保存有芽的图，或者你可以全保存)
        if has_sprout:
            count_sprouted += 1
            if img.shape[2] == 4:
                b, g, r, a = cv2.split(img)
            else:
                b, g, r = cv2.split(img)

            # 保存只有芽的透明PNG
            sprout_rgba = cv2.merge([b, g, r, sprout_mask])
            save_p_sprout = os.path.join(OUTPUT_ROOT, f"{base_name}_sprout.png")
            os.makedirs(os.path.dirname(save_p_sprout), exist_ok=True)
            cv2.imwrite(save_p_sprout, sprout_rgba)

    print(f"处理完成！总计 {count_total} 张，识别出发芽 {count_sprouted} 张。")


if __name__ == "__main__":
    main()