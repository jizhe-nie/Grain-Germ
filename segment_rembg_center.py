import os
import glob
import cv2
import numpy as np
from rembg import remove, new_session
from tqdm import tqdm
import onnxruntime as ort

# ================= 配置区域 =================
INPUT_ROOT = "output/results_seeds"
OUTPUT_ROOT = "output/segmented_seeds_crop_fix"

# 1. 裁剪比例：0.15 表示只保留图片中心 70% 的区域 (四周各切掉 15%)
# 这能物理移除 000.jpg, 069.jpg 的栅栏
CROP_RATIO = 0.15

# 2. 对比度增强强度 (CLAHE Clip Limit)
# 值越大，种子越亮，根系越清晰。建议 3.0 - 4.0
CLAHE_LIMIT = 4.0

# 模型选择: isnet-general-use 对细微结构(根)更好，u2net 更通用
MODEL_NAME = "isnet-general-use"


# ===========================================

def apply_clahe(image):
    """
    预处理：不压暗，而是提亮！
    让黑背景保持黑，但让种子和根变得极亮，强迫 AI 关注它。
    """
    # 转 Lab 色彩空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 应用 CLAHE (局部自适应直方图均衡)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_LIMIT, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # 合并回 BGR
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def process_single_image(img_path, session):
    # 1. 读取原图
    original_img = cv2.imread(img_path)
    if original_img is None: return None
    h, w = original_img.shape[:2]

    # ==========================================
    # 步骤 A: 物理裁剪 (Center Crop)
    # 直接切掉周围的干扰，而不是涂黑，防止 AI 识别出矩形框
    # ==========================================
    margin_h = int(h * CROP_RATIO)
    margin_w = int(w * CROP_RATIO)

    # 提取中心 ROI (Region of Interest)
    # 种子一定在这里
    crop_img = original_img[margin_h:h - margin_h, margin_w:w - margin_w]

    if crop_img.size == 0: return None  # 防止切空

    # ==========================================
    # 步骤 B: 提亮增强 (CLAHE)
    # ==========================================
    # 增强对比度，让种子从黑水中“跳”出来
    input_crop = apply_clahe(crop_img)

    # ==========================================
    # 步骤 C: AI 推理 (仅针对中心部分)
    # ==========================================
    _, enc_img = cv2.imencode(".jpg", input_crop)
    output_bytes = remove(enc_img.tobytes(), session=session)

    # 解码结果
    nparr = np.frombuffer(output_bytes, np.uint8)
    crop_result_rgba = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # 提取 Mask (Alpha)
    crop_alpha = crop_result_rgba[:, :, 3]

    # ==========================================
    # 步骤 D: 原位贴回 (Paste Back)
    # ==========================================
    # 创建一张全黑的大图 (全透明)
    final_alpha = np.zeros((h, w), dtype=np.uint8)

    # 把处理好的中心 Mask 贴回去
    final_alpha[margin_h:h - margin_h, margin_w:w - margin_w] = crop_alpha

    # 最后的合成：用【原图】+【合成Mask】
    # 这样既去掉了背景，又保留了原始种子的颜色
    b, g, r = cv2.split(original_img)
    final_rgba = cv2.merge([b, g, r, final_alpha])

    return final_rgba


def main():
    # 检测 GPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print("✅ GPU 加速已启用")
    else:
        print("⚠️ 使用 CPU 运行")

    print(f"加载模型: {MODEL_NAME}...")
    session = new_session(model_name=MODEL_NAME, providers=providers)

    all_images = glob.glob(os.path.join(INPUT_ROOT, "**", "*.jpg"), recursive=True)
    print(f"开始处理 {len(all_images)} 张图片 (裁剪-提亮-复原策略)...")

    for img_path in tqdm(all_images):
        try:
            result_rgba = process_single_image(img_path, session)

            if result_rgba is not None:
                # 保存
                relative_path = os.path.relpath(img_path, INPUT_ROOT)
                save_path = os.path.join(OUTPUT_ROOT, relative_path)
                save_path = save_path.replace(".jpg", ".png")

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, result_rgba)

        except Exception as e:
            print(f"Error: {img_path} - {e}")


if __name__ == "__main__":
    main()