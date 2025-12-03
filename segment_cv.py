import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

# ================= 配置 =================
INPUT_ROOT = "results_seeds"  # 上一步生成的文件夹
OUTPUT_ROOT = "output/segmented_seeds"  # 结果输出文件夹
DEBUG_MODE = False  # 设为True会保存黑白Mask以便调试


# =======================================

def process_segmentation(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None: return

    # 1. 预处理：转灰度 + 高斯模糊 (去噪)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. 阈值分割
    # 使用 Otsu 自动寻找最佳阈值。
    # 因为背景是黑的，前景是亮的，THRESH_BINARY 即可。
    # 针对部分特别暗的根系，可能需要适当调低阈值，或者使用自适应阈值
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 形态学操作 (关键步骤)
    # kernel大小决定了去除噪点的能力。(3,3) 适合去除细小反光
    kernel = np.ones((3, 3), np.uint8)

    # 开运算：先腐蚀后膨胀 -> 去除孤立的小白点（水面反光）
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    # 膨胀：让种子稍微胖一点，填补内部空洞，并把细根连起来
    dilation = cv2.dilate(opening, kernel, iterations=1)

    # 4. 轮廓查找与筛选 (中心约束法)
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return  # 没找到轮廓

    h, w = img.shape[:2]
    img_center = (w // 2, h // 2)

    best_cnt = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # 过滤1: 面积太小的噪点
        if area < 100: continue

        # 过滤2: 检查轮廓是否贴边 (去除切图残留的白色隔板)
        x, y, cw, ch = cv2.boundingRect(cnt)
        # 如果轮廓紧贴图片四个边缘的任何一个，大概率是隔板
        margin = 2
        if x < margin or y < margin or (x + cw) > w - margin or (y + ch) > h - margin:
            # 但要小心种子长大后也会贴边，所以加一个判定：
            # 如果面积非常大(例如占画面50%以上)，那可能是长大后的种子，不过滤
            if area < (h * w * 0.5):
                continue

        # 策略: 找最大的轮廓，通常就是种子
        if area > max_area:
            max_area = area
            best_cnt = cnt

    # 5. 生成结果
    if best_cnt is not None:
        # 创建最终掩膜
        final_mask = np.zeros_like(gray)
        cv2.drawContours(final_mask, [best_cnt], -1, 255, -1)

        # 方案A: 保存为透明背景的PNG (推荐，方便后续叠图)
        b, g, r = cv2.split(img)
        rgba = cv2.merge([b, g, r, final_mask])
        cv2.imwrite(save_path.replace(".jpg", ".png"), rgba)

        # 方案B: 如果你需要保留原文件名且只要黑色背景 JPG
        # result = cv2.bitwise_and(img, img, mask=final_mask)
        # cv2.imwrite(save_path, result)
    else:
        # 如果筛选完没有轮廓，说明图片可能是全黑或只有噪点
        # 这种情况下可以复制原图或者存一个全黑图
        pass


def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    # 获取所有种子文件夹
    seed_dirs = sorted(glob.glob(os.path.join(INPUT_ROOT, "seed_*")))

    print(f"开始处理 {len(seed_dirs)} 个种子文件夹...")

    for s_dir in tqdm(seed_dirs):
        dir_name = os.path.basename(s_dir)  # e.g., seed_01
        target_dir = os.path.join(OUTPUT_ROOT, dir_name)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # 处理该种子下的所有图片
        imgs = sorted(glob.glob(os.path.join(s_dir, "*.jpg")))
        for img_p in imgs:
            fname = os.path.basename(img_p)
            save_p = os.path.join(target_dir, fname)
            process_segmentation(img_p, save_p)

    print("分割完成！")


if __name__ == "__main__":
    main()