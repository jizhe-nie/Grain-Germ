import cv2
import numpy as np
import os
import glob
import torch
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_ROOT = "output/results_seeds"
OUTPUT_ROOT = "output/segmented_seeds_reconstruct"  # 新的输出目录
CHECKPOINT_PATH = "../weight/sam_vit_l_0b3195.pth"
MODEL_TYPE = "vit_l"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================================

def apply_clahe(image):
    """强力增强对比度，专门为了提取隐约的芽"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # ClipLimit 设高一点，让芽更明显
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def get_loose_mask(image):
    """
    生成一张“高灵敏度”的候选Mask
    这里面会包含芽，但也会包含很多背景噪点。
    """
    # 1. 强增强
    enhanced = apply_clahe(image)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. 自动阈值 (Otsu)
    # 如果芽特别暗，可能需要将阈值系数调低，例如 0.8 * otsu_val
    otsu_val, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 稍微膨胀一点，确保芽和种子之间没有断裂
    kernel = np.ones((3, 3), np.uint8)
    loose_mask = cv2.dilate(binary, kernel, iterations=1)

    return loose_mask


def find_seed_point(image):
    # (保持之前的智能找点逻辑不变，这是给SAM用的)
    h, w = image.shape[:2]
    img_center = np.array([w // 2, h // 2])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_point = np.array([[w // 2, h // 2]])
    min_dist = float('inf')

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50: continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        margin = 2
        if x <= margin or y <= margin or (x + cw) >= w - margin or (y + ch) >= h - margin:
            if area < (h * w * 0.6): continue

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist = np.linalg.norm(np.array([cx, cy]) - img_center)
            if dist < min_dist:
                min_dist = dist
                best_point = np.array([[cx, cy]])
    return best_point


def main():
    print(f"加载模型: {MODEL_TYPE} ({DEVICE})...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    all_imgs = sorted(glob.glob(os.path.join(INPUT_ROOT, "**", "*.jpg"), recursive=True))
    print(f"开始处理 {len(all_imgs)} 张图片 (启用形态学重建策略)...")

    for img_path in tqdm(all_imgs):
        original_img = cv2.imread(img_path)
        if original_img is None: continue

        # ==========================================
        # 步骤 1: 获得 SAM 的“干净” Mask (锚点)
        # ==========================================
        input_point = find_seed_point(original_img)
        enhanced_img = apply_clahe(original_img)  # 给SAM看增强图
        enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

        try:
            predictor.set_image(enhanced_rgb)
            with torch.inference_mode():
                # 只有 CUDA 环境才需要 autocast
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

            # 【关键修改】只取分数最高的！绝对不要取 Masks[2]
            best_idx = np.argmax(scores)
            sam_mask = (masks[best_idx] * 255).astype(np.uint8)

            # ==========================================
            # 步骤 2: 获得“宽松”的候选 Mask
            # ==========================================
            # 这个mask包含了芽，但也包含了背景反光
            loose_mask = get_loose_mask(original_img)

            # ==========================================
            # 步骤 3: 形态学重建 (只保留相连部分)
            # ==========================================
            # 逻辑：找出 loose_mask 中所有的连通域，
            # 只有那些与 sam_mask 有重叠(交集)的连通域，才被保留。

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(loose_mask, connectivity=8)

            final_mask = np.zeros_like(loose_mask)

            # 获取 SAM Mask 覆盖区域的 label 索引
            # 我们看 sam_mask 为 255 的地方，在 labels 里对应什么数字
            overlap_labels = np.unique(labels[sam_mask == 255])

            # 遍历所有重叠的 label ID
            for label_id in overlap_labels:
                if label_id == 0: continue  # 背景ID忽略

                # 将该连通域加入最终结果
                final_mask[labels == label_id] = 255

            # 【保底策略】
            # 如果 loose_mask 阈值没切好，导致 final_mask 比 sam_mask 还小
            # 我们至少要保留 sam_mask (保证谷粒不丢)
            final_mask = cv2.bitwise_or(final_mask, sam_mask)

            # ==========================================
            # 保存
            # ==========================================
            b, g, r = cv2.split(original_img)
            rgba = cv2.merge([b, g, r, final_mask])

            relative_path = os.path.relpath(img_path, INPUT_ROOT)
            save_path = os.path.join(OUTPUT_ROOT, relative_path)
            save_path = save_path.replace(".jpg", ".png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, rgba)

        except Exception as e:
            print(f"Error: {img_path} - {e}")
            if DEVICE == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__":
    main()