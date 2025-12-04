import os
import glob
import cv2
import numpy as np
from rembg import remove, new_session
from tqdm import tqdm
import onnxruntime as ort

# ================= 配置区域 =================
# 请确保这个路径下有你的图片 (比如 seed_01/xxx.jpg)
INPUT_ROOT = "output/results_seeds"
OUTPUT_ROOT = "output/segmented_seeds_perfect_fix"

# 1. 裁剪比例 0.15 = 切掉四周 15% (物理移除栅栏)
CROP_RATIO = 0.15

# 2. 对比度增强强度 (提亮种子)
CLAHE_LIMIT = 4.0

# 3. 模型选择
MODEL_NAME = "isnet-general-use"


# ===========================================

def apply_clahe(image):
    """预处理：局部提亮"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_LIMIT, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def post_process_cleanup(rgba_image, original_w, original_h):
    """
    【二次清洗】去除 rembg 可能残留的孤立噪点
    """
    # 提取 Alpha 通道
    alpha = rgba_image[:, :, 3]

    # 二值化
    _, binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1: return rgba_image  # 只有背景

    # 寻找最佳连通域 (面积大 + 离中心近)
    img_center = np.array([original_w // 2, original_h // 2])
    best_label = -1
    max_score = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 50: continue  # 忽略极小噪点

        cx, cy = centroids[i]
        dist = np.linalg.norm(np.array([cx, cy]) - img_center)

        # 距离惩罚：如果离中心太远，视为边缘垃圾，直接丢弃
        if dist > (min(original_w, original_h) * 0.4):
            score = 0
        else:
            score = area

        if score > max_score:
            max_score = score
            best_label = i

    # 重绘 Mask
    new_alpha = np.zeros_like(alpha)
    if best_label != -1:
        new_alpha[labels == best_label] = 255
    else:
        return np.zeros_like(rgba_image)  # 没找到有效种子

    # 更新 RGBA
    cleaned_rgba = rgba_image.copy()
    cleaned_rgba[:, :, 3] = new_alpha

    return cleaned_rgba


def process_single_image(img_path, session):
    original_img = cv2.imread(img_path)
    if original_img is None: return None
    h, w = original_img.shape[:2]

    # --- Step 1: 物理裁剪 (去除边缘栅栏) ---
    margin_h = int(h * CROP_RATIO)
    margin_w = int(w * CROP_RATIO)
    # 提取中心
    crop_img = original_img[margin_h:h - margin_h, margin_w:w - margin_w]
    if crop_img.size == 0: return None

    # --- Step 2: 提亮增强 (让种子从黑水中显形) ---
    input_crop = apply_clahe(crop_img)

    # --- Step 3: AI 推理 (rembg) ---
    _, enc_img = cv2.imencode(".jpg", input_crop)
    output_bytes = remove(enc_img.tobytes(), session=session)
    nparr = np.frombuffer(output_bytes, np.uint8)
    crop_result_rgba = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if crop_result_rgba is None: return None

    # --- Step 4: 原位贴回 (恢复原尺寸) ---
    crop_alpha = crop_result_rgba[:, :, 3]
    final_alpha = np.zeros((h, w), dtype=np.uint8)
    final_alpha[margin_h:h - margin_h, margin_w:w - margin_w] = crop_alpha

    # 合成：用原图 + 新Mask
    b, g, r = cv2.split(original_img)
    rough_rgba = cv2.merge([b, g, r, final_alpha])

    # --- Step 5: 二次清洗 (去除残留反光点) ---
    final_clean_rgba = post_process_cleanup(rough_rgba, w, h)

    return final_clean_rgba


def main():
    # 1. 硬件加速检测 (适配 Mac M系列)
    available_providers = ort.get_available_providers()
    providers = []

    if 'CoreMLExecutionProvider' in available_providers:
        print("🍎 检测到 Mac Apple Silicon (M系列芯片)，已启用 CoreML 加速")
        providers.append('CoreMLExecutionProvider')
    elif 'CUDAExecutionProvider' in available_providers:
        print("✅ 检测到 NVIDIA GPU，已启用 CUDA 加速")
        providers.append('CUDAExecutionProvider')
    else:
        print("⚠️ 未检测到专用加速器，将使用 CPU 运行")

    providers.append('CPUExecutionProvider')

    # 2. 加载模型
    print(f"加载模型: {MODEL_NAME}...")
    session = new_session(model_name=MODEL_NAME, providers=providers)

    # 3. 扫描文件
    # 注意：请检查 INPUT_ROOT 是否正确
    search_path = os.path.join(INPUT_ROOT, "**", "*.jpg")
    all_images = glob.glob(search_path, recursive=True)

    print(f"在 {INPUT_ROOT} 下检测到 {len(all_images)} 张图片，开始处理...")

    if len(all_images) == 0:
        print("❌ 未找到图片！请检查代码顶部的 INPUT_ROOT 路径是否正确。")
        return

    # 4. 循环处理
    for img_path in tqdm(all_images):
        try:
            result_rgba = process_single_image(img_path, session)

            if result_rgba is not None:
                # 保持原目录结构保存
                relative_path = os.path.relpath(img_path, INPUT_ROOT)
                save_path = os.path.join(OUTPUT_ROOT, relative_path)
                save_path = save_path.replace(".jpg", ".png")

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, result_rgba)

        except Exception as e:
            print(f"Error: {img_path} - {e}")


if __name__ == "__main__":
    main()