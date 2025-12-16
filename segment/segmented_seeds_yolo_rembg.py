import os
import glob
import cv2
import numpy as np
from rembg import remove, new_session
from ultralytics import YOLO
from tqdm import tqdm
import onnxruntime as ort

# ================= 配置区域 =================
INPUT_ROOT = "../output/results_seeds"
OUTPUT_ROOT = "../output/segmented_seeds_yolo_rembg"
WEIGHTS_PATH = "../weight/best.pt"  # 你的 YOLOv8n 模型路径

# 1. 扩边比例 (Padding)
# YOLO 框通常很贴合谷粒，为了包含发出的芽，我们需要把框往外扩一圈
# 0.2 表示向四周各扩展宽高的 20%
EXPAND_RATIO = 0.3

# 2. 对比度增强 (CLAHE)
# 在送给 rembg 之前提亮，帮助识别半透明根系
CLAHE_LIMIT = 4.0

# 3. rembg 模型
MODEL_NAME = "isnet-general-use"


# ===========================================

def apply_clahe(image):
    """局部提亮增强"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_LIMIT, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def get_yolo_box(model, image):
    """
    使用 YOLO 检测谷粒，并返回扩展后的裁剪坐标
    """
    h, w = image.shape[:2]

    # 运行推理 (conf=0.25 过滤低置信度)
    results = model(image, verbose=False, conf=0.25)

    # 如果没检测到 (比如图片全黑)，返回 None
    if not results or len(results[0].boxes) == 0:
        return None

    # 获取置信度最高的那个框
    # boxes.data 格式: [x1, y1, x2, y2, conf, cls]
    best_box = results[0].boxes.data[0].cpu().numpy()
    x1, y1, x2, y2 = best_box[:4]

    # --- 关键步骤：动态扩边 (Padding) ---
    # 芽可能会长出 YOLO 的框外，所以必须扩
    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * EXPAND_RATIO)
    pad_y = int(box_h * EXPAND_RATIO)

    # 修正坐标，防止越出图片边界
    new_x1 = int(max(0, x1 - pad_x))
    new_y1 = int(max(0, y1 - pad_y))
    new_x2 = int(min(w, x2 + pad_x))
    new_y2 = int(min(h, y2 + pad_y))

    return (new_x1, new_y1, new_x2, new_y2)


def main():
    # 1. 加载 YOLO 模型
    print(f"正在加载 YOLO 模型: {WEIGHTS_PATH} ...")
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ 错误：找不到模型文件 {WEIGHTS_PATH}")
        return
    yolo_model = YOLO(WEIGHTS_PATH)

    # 2. 加载 rembg Session (配置 Mac 加速)
    available_providers = ort.get_available_providers()
    providers = []
    if 'CoreMLExecutionProvider' in available_providers:
        print("🍎 Mac M芯片加速已启用 (CoreML)")
        providers.append('CoreMLExecutionProvider')
    elif 'CUDAExecutionProvider' in available_providers:
        print("✅ NVIDIA GPU 加速已启用")
        providers.append('CUDAExecutionProvider')
    else:
        providers.append('CPUExecutionProvider')

    rembg_session = new_session(model_name=MODEL_NAME, providers=providers)

    # 3. 开始处理
    all_images = glob.glob(os.path.join(INPUT_ROOT, "**", "*.jpg"), recursive=True)
    print(f"开始处理 {len(all_images)} 张图片 (YOLO定位 -> 扩边 -> rembg分割)...")

    for img_path in tqdm(all_images):
        try:
            original_img = cv2.imread(img_path)
            if original_img is None: continue
            h, w = original_img.shape[:2]

            # --- Step A: YOLO 检测 ---
            crop_coords = get_yolo_box(yolo_model, original_img)

            if crop_coords is None:
                # 如果 YOLO 没检测到，说明图里可能没种子，或者太暗
                # 输出全黑图，跳过后续
                final_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            else:
                x1, y1, x2, y2 = crop_coords

                # 裁剪出谷粒区域
                crop_img = original_img[y1:y2, x1:x2]

                if crop_img.size == 0: continue

                # --- Step B: rembg 分割 ---
                # 预处理提亮
                enhanced_crop = apply_clahe(crop_img)

                # rembg 推理
                _, enc_img = cv2.imencode(".jpg", enhanced_crop)
                output_bytes = remove(enc_img.tobytes(), session=rembg_session)
                nparr = np.frombuffer(output_bytes, np.uint8)
                crop_result_rgba = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

                # --- Step C: 原位贴回 (Restoration) ---
                # 这一步保证你的输出图片还是 1920x1080 (或其他原尺寸)，且位置正确

                # 创建全透明大图
                final_rgba = np.zeros((h, w, 4), dtype=np.uint8)

                # 提取 crop 的 mask
                crop_alpha = crop_result_rgba[:, :, 3]

                # 将 mask 贴回到大图的对应坐标位置
                final_alpha = np.zeros((h, w), dtype=np.uint8)
                final_alpha[y1:y2, x1:x2] = crop_alpha

                # 合成最终结果 (使用原图 RGB)
                b, g, r = cv2.split(original_img)
                final_rgba = cv2.merge([b, g, r, final_alpha])

            # 保存
            relative_path = os.path.relpath(img_path, INPUT_ROOT)
            save_path = os.path.join(OUTPUT_ROOT, relative_path)
            save_path = save_path.replace(".jpg", ".png")

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, final_rgba)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    main()