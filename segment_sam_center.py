import cv2
import numpy as np
import os
import glob
import torch
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_ROOT = "output/results_seeds"
OUTPUT_ROOT = "output/segmented_seeds_smart_v2"
CHECKPOINT_PATH = "weight/sam_vit_l_0b3195.pth"  # 必须用 ViT-L
MODEL_TYPE = "vit_l"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 调试开关
DEBUG_MODE = True
DEBUG_DIR = "output/debug_smart_v2"


# ===========================================

def get_strict_center_point(image):
    """
    【极度严格】只在图片最中心的 40% 区域寻找最亮物体
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. 定义严格的中心区域 (Inner 40%)
    # 比如 200x200 的图，只看中间 80x80 的区域
    margin_x = int(w * 0.30)
    margin_y = int(h * 0.30)

    # 构造 Mask，只保留中心
    center_mask = np.zeros_like(gray)
    center_mask[margin_y:h - margin_y, margin_x:w - margin_x] = 255

    masked_gray = cv2.bitwise_and(gray, gray, mask=center_mask)

    # 2. 寻找最亮区域
    # 使用高斯模糊 + Otsu
    blurred = cv2.GaussianBlur(masked_gray, (5, 5), 0)
    # 只有中心区域参与阈值计算，避免边缘栅栏干扰阈值
    center_pixels = blurred[margin_y:h - margin_y, margin_x:w - margin_x]
    if center_pixels.size == 0: return None, False

    # 计算局部阈值
    otsu_val, _ = cv2.threshold(center_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 应用到全图 (mask外虽然是0，但不会有影响)
    _, binary = cv2.threshold(blurred, otsu_val, 255, cv2.THRESH_BINARY)

    # 3. 找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # 如果中心什么都没有（比如全黑），直接返回中心坐标作为保底
        # 但标记为 False，表示不太可信
        return np.array([[w // 2, h // 2]]), False

    # 找最大的那个（在中心区域内的最大物体）
    best_cnt = max(contours, key=cv2.contourArea)

    M = cv2.moments(best_cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return np.array([[cx, cy]]), True

    return np.array([[w // 2, h // 2]]), False


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def select_best_mask(masks, scores, img_center):
    """
    【智能选择器】不看分数，看位置！
    选择重心离图片中心最近的那个 Mask。
    """
    best_idx = 0
    min_dist = float('inf')

    cx_img, cy_img = img_center

    for i, mask in enumerate(masks):
        # 计算 Mask 重心
        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] == 0: continue

        mcx = int(M["m10"] / M["m00"])
        mcy = int(M["m01"] / M["m00"])

        # 距离中心的距离
        dist = np.sqrt((mcx - cx_img) ** 2 + (mcy - cy_img) ** 2)

        # 惩罚项：如果 mask 面积太大（覆盖全图），给它加 huge penalty
        area_ratio = M["m00"] / (mask.shape[0] * mask.shape[1])
        if area_ratio > 0.8:  # 80% 都是前景，肯定是背景误判
            dist += 1000

        if dist < min_dist:
            min_dist = dist
            best_idx = i

    return best_idx


def main():
    print(f"加载 ViT-L 模型 ({DEVICE})...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if DEBUG_MODE:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    all_imgs = sorted(glob.glob(os.path.join(INPUT_ROOT, "**", "*.jpg"), recursive=True))
    print(f"处理 {len(all_imgs)} 张图片 (Smart Selector + 强力切割)...")

    for img_path in tqdm(all_imgs):
        original_img = cv2.imread(img_path)
        if original_img is None: continue
        h, w = original_img.shape[:2]

        # 1. 严格找点
        input_point, is_found = get_strict_center_point(original_img)

        # 2. SAM 预测
        enhanced_img = apply_clahe(original_img)
        enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

        try:
            predictor.set_image(enhanced_rgb)
            with torch.inference_mode():
                # 兼容性写法
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

            # 3. 【核心修正】智能选择最佳 Mask
            # 放弃 argmax(scores)，改用距离中心最近的原则
            best_idx = select_best_mask(masks, scores, (w // 2, h // 2))
            raw_mask = (masks[best_idx] * 255).astype(np.uint8)

            # 4. 【狠辣切割】护城河策略
            # 根据你的000.jpg，栅栏很宽。我们切掉四周 15 像素
            # 这能物理断开 070.jpg 那样的粘连
            border_cut = 15
            moat_mask = np.zeros_like(raw_mask)
            moat_mask[border_cut:h - border_cut, border_cut:w - border_cut] = 255

            final_mask = cv2.bitwise_and(raw_mask, moat_mask)

            # 5. 连通域过滤
            # 只保留包含 prompt 点的那个区域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask)

            # 修正坐标防止越界
            py, px = input_point[0][1], input_point[0][0]
            py = min(max(py, 0), h - 1)
            px = min(max(px, 0), w - 1)

            target_id = labels[py, px]

            clean_mask = np.zeros_like(final_mask)

            if target_id != 0:
                clean_mask[labels == target_id] = 255
            else:
                # 【重要修正】如果中心点都没选中，说明这次分割彻底失败（点在背景上）
                # 此时绝对不要选最大面积（那是栅栏！）
                # 宁愿输出全黑，也不要错误数据
                # 如果你想抢救一下，可以尝试找离中心最近的那个连通域（而不是最大的）

                # 寻找距离中心最近的组件
                min_comp_dist = float('inf')
                best_comp_id = -1

                for i in range(1, num_labels):  # 跳过背景0
                    # 组件重心
                    ccx, ccy = centroids[i]
                    dist = np.sqrt((ccx - w // 2) ** 2 + (ccy - h // 2) ** 2)
                    if dist < min_comp_dist:
                        min_comp_dist = dist
                        best_comp_id = i

                # 只有当这个最近的组件真的比较近（比如在中心区域内）才保留
                if best_comp_id != -1 and min_comp_dist < min(w, h) * 0.25:
                    clean_mask[labels == best_comp_id] = 255
                else:
                    # 放弃治疗，输出全黑。这比输出错误的栅栏要好得多。
                    pass

                    # Debug 保存
            if DEBUG_MODE:
                # 画出点的位置和最终mask轮廓
                vis = original_img.copy()
                cv2.circle(vis, (px, py), 3, (0, 0, 255), -1)

                # 画边缘切割线
                cv2.rectangle(vis, (border_cut, border_cut), (w - border_cut, h - border_cut), (0, 255, 255), 1)

                # 叠加 mask 轮廓
                contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)

                rel_name = os.path.relpath(img_path, INPUT_ROOT).replace("/", "_")
                cv2.imwrite(os.path.join(DEBUG_DIR, rel_name), vis)

            # 保存最终结果
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