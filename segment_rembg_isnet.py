import cv2
import numpy as np
import os
import glob
from rembg import remove, new_session
from tqdm import tqdm
import onnxruntime as ort  # 用于检查GPU是否可用

# ================= 配置区域 =================
INPUT_ROOT = "output/results_seeds"
OUTPUT_ROOT = "output/segmented_seeds_rembg_final"
MODEL_NAME = "isnet-general-use"
BORDER_CUT = 20


# ===========================================

def apply_strong_clahe(image):
    # 预处理：增强对比度
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def post_process_mask(alpha_mask, original_h, original_w):
    # 后处理：物理切割 + 中心优选
    safe_zone = np.zeros_like(alpha_mask)
    safe_zone[BORDER_CUT:original_h - BORDER_CUT, BORDER_CUT:original_w - BORDER_CUT] = 255
    cut_mask = cv2.bitwise_and(alpha_mask, safe_zone)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cut_mask, connectivity=8)

    if num_labels <= 1: return None

    best_label = -1
    min_dist_to_center = float('inf')
    center_img = np.array([original_w // 2, original_h // 2])

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 50: continue

        cx, cy = centroids[i]
        dist = np.linalg.norm(np.array([cx, cy]) - center_img)

        if dist < min_dist_to_center:
            min_dist_to_center = dist
            best_label = i

    if min_dist_to_center > (min(original_h, original_w) * 0.4):
        return None

    final_mask = np.zeros_like(cut_mask)
    final_mask[labels == best_label] = 255
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    return final_mask


def main():
    # 检查 GPU 是否可用
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print("✅ 成功检测到 NVIDIA GPU，正在启用 CUDA 加速！")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        print("⚠️ 未检测到 GPU 或 CUDA 配置不正确，将使用 CPU 运行 (速度较慢)。")
        providers = ['CPUExecutionProvider']

    print(f"加载 rembg 模型: {MODEL_NAME}...")
    # 显式传入 providers 参数
    session = new_session(model_name=MODEL_NAME, providers=providers)

    all_imgs = sorted(glob.glob(os.path.join(INPUT_ROOT, "**", "*.jpg"), recursive=True))
    print(f"开始处理 {len(all_imgs)} 张图片...")

    for img_path in tqdm(all_imgs):
        try:
            original_img = cv2.imread(img_path)
            if original_img is None: continue
            h, w = original_img.shape[:2]

            enhanced_img = apply_strong_clahe(original_img)
            _, enc_img = cv2.imencode(".jpg", enhanced_img)
            img_bytes = enc_img.tobytes()

            # 推理
            output_bytes = remove(img_bytes, session=session)

            nparr = np.frombuffer(output_bytes, np.uint8)
            rembg_result = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

            if rembg_result is None: continue  # 防止解码失败
            alpha_channel = rembg_result[:, :, 3]

            clean_mask = post_process_mask(alpha_channel, h, w)

            if clean_mask is None:
                final_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            else:
                b, g, r = cv2.split(original_img)
                final_rgba = cv2.merge([b, g, r, clean_mask])

            relative_path = os.path.relpath(img_path, INPUT_ROOT)
            save_path = os.path.join(OUTPUT_ROOT, relative_path)
            save_path = save_path.replace(".jpg", ".png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, final_rgba)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    main()