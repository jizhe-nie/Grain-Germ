import cv2
import numpy as np
import os
import glob
import torch
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_ROOT = "output/results_seeds"
OUTPUT_ROOT = "output/segmented_seeds_hsv_fix"
CHECKPOINT_PATH = "../weight/sam_vit_l_0b3195.pth"  # 继续用 ViT-L
MODEL_TYPE = "vit_l"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 调试模式：开启后会保存一张 visualize.jpg 给你看它到底锁定了哪里
DEBUG_MODE = True
DEBUG_DIR = "output/debug_hsv_points"


# ===========================================

def get_hsv_seed_point(image):
    """
    【核心改进】利用颜色来区分种子和栅栏
    种子 = 有颜色 (Saturation > 阈值)
    栅栏 = 黑白灰 (Saturation < 阈值)
    """
    h, w = image.shape[:2]
    img_center = np.array([w // 2, h // 2])

    # 转到 HSV 空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # ========================================================
    # 策略 1: 饱和度过滤 (关键!)
    # 栅栏是白色的，S值很低；种子是黄色的，S值较高。
    # 阈值设为 25 (0-255范围)，能过滤掉大部分白光和栅栏
    # ========================================================
    _, s_mask = cv2.threshold(S, 30, 255, cv2.THRESH_BINARY)

    # 策略 2: 亮度过滤 (保留亮的，去掉背景黑水)
    _, v_mask = cv2.threshold(V, 40, 255, cv2.THRESH_BINARY)

    # 结合两者：既要有亮度，又要有颜色
    valid_mask = cv2.bitwise_and(s_mask, v_mask)

    # 形态学去噪
    kernel = np.ones((3, 3), np.uint8)
    valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 查找轮廓
    contours, _ = cv2.findContours(valid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_point = np.array([[w // 2, h // 2]])
    max_score = 0
    found = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30: continue

        # 计算重心
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 计算距离中心的权重 (越近越好)
            dist = np.linalg.norm(np.array([cx, cy]) - img_center)
            score = area / (dist + 1)  # 面积大且离中心近的得分高

            if score > max_score:
                max_score = score
                best_point = np.array([[cx, cy]])
                found = True

    return best_point, found, valid_mask


def apply_clahe(image):
    """增强对比度"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def main():
    print(f"加载模型: {MODEL_TYPE} ({DEVICE})...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if DEBUG_MODE:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    all_imgs = sorted(glob.glob(os.path.join(INPUT_ROOT, "**", "*.jpg"), recursive=True))
    print(f"处理 {len(all_imgs)} 张图片 (HSV定位 + 边缘切割)...")

    for img_path in tqdm(all_imgs):
        original_img = cv2.imread(img_path)
        if original_img is None: continue
        h, w = original_img.shape[:2]

        # 1. 使用 HSV 找点
        input_point, is_found, debug_mask = get_hsv_seed_point(original_img)

        # [调试] 如果没找到点，或者开启了调试模式，保存图片看看
        if DEBUG_MODE and is_found:
            vis = original_img.copy()
            cv2.circle(vis, (input_point[0][0], input_point[0][1]), 4, (0, 0, 255), -1)
            # 把 HSV mask 叠上去方便看
            mask_vis = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)
            vis = np.hstack([vis, mask_vis])

            # 保持相对路径保存
            rel_name = os.path.relpath(img_path, INPUT_ROOT).replace("/", "_")
            cv2.imwrite(os.path.join(DEBUG_DIR, rel_name), vis)

        # 2. SAM 预测
        enhanced_img = apply_clahe(original_img)
        enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

        try:
            predictor.set_image(enhanced_rgb)
            with torch.inference_mode():
                # 根据设备选择是否启用 autocast
                if DEVICE == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        masks, scores, _ = predictor.predict(
                            point_coords=input_point,
                            point_labels=np.array([1]),
                            multimask_output=True,
                        )
                else:
                    masks, scores, _ = predictor.predict(
                        point_coords=input_point,
                        point_labels=np.array([1]),
                        multimask_output=True,
                    )

            # 3. 选择 Mask
            # 依然选分数最高的，因为现在 Prompt 点非常准，SAM 不太会选错
            best_idx = np.argmax(scores)
            raw_mask = (masks[best_idx] * 255).astype(np.uint8)

            # ========================================================
            # 策略 3: 强制边缘切割 (Kill the Grid)
            # ========================================================
            # 创建一个比图片小一圈的矩形 Mask
            border_crop = 10  # 剪掉四周 10 个像素
            safe_zone = np.zeros_like(raw_mask)
            safe_zone[border_crop:h - border_crop, border_crop:w - border_crop] = 255

            # 只要超出这个安全区的 mask，全部砍掉
            # 这能物理切断连在栅栏上的像素
            final_mask = cv2.bitwise_and(raw_mask, safe_zone)

            # 再次做连通域分析，防止切断后留下了孤立的小块
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask)

            # 找到 Prompt 点所在的区域
            py, px = input_point[0][1], input_point[0][0]
            # 修正坐标防止越界
            py = min(max(py, 0), h - 1)
            px = min(max(px, 0), w - 1)

            target_id = labels[py, px]

            clean_mask = np.zeros_like(final_mask)
            if target_id != 0:
                clean_mask[labels == target_id] = 255
            else:
                # 备用：找最大
                if len(stats) > 1:
                    max_id = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                    clean_mask[labels == max_id] = 255

            # 保存
            b, g, r = cv2.split(original_img)
            rgba = cv2.merge([b, g, r, clean_mask])

            relative_path = os.path.relpath(img_path, INPUT_ROOT)
            save_path = os.path.join(OUTPUT_ROOT, relative_path)
            save_path = save_path.replace(".jpg", ".png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, rgba)

        except Exception as e:
            print(f"Error: {e}")
            if DEVICE == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__":
    main()