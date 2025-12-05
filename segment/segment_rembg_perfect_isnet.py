import os
import glob
import cv2
import numpy as np
from rembg import remove, new_session
from tqdm import tqdm
import onnxruntime as ort

# ================= 配置区域 =================
INPUT_ROOT = "../output/results_seeds"
OUTPUT_ROOT = "../output/segmented_seeds_perfect_isnet"

# 1. 裁剪比例 (保持之前的成功参数)
CROP_RATIO = 0.1

# 2. 对比度增强强度
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
    【新增】战场打扫逻辑
    对 rembg 的结果进行二次清洗，去除残留的噪点和边缘碎片。
    """
    # 提取 Alpha 通道
    alpha = rgba_image[:, :, 3]

    # 1. 二值化 (确保 mask 是黑白的)
    _, binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

    # 2. 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:  # 全黑，没东西
        return rgba_image  # 返回原样或全黑均可

    # 3. 寻找“真正的种子”
    # 规则：面积最大，且距离图片中心不要太远
    img_center = np.array([original_w // 2, original_h // 2])
    best_label = -1
    max_score = 0

    for i in range(1, num_labels):  # 跳过背景0
        area = stats[i, cv2.CC_STAT_AREA]

        # 过滤极小噪点 (比如小于 100 像素的碎渣)
        if area < 100: continue

        # 计算重心距离
        cx, cy = centroids[i]
        dist = np.linalg.norm(np.array([cx, cy]) - img_center)

        # 评分：面积大 + 离中心近
        # 距离惩罚系数：如果距离超过图片半径的 40%，得分急剧下降
        if dist > (min(original_w, original_h) * 0.4):
            score = 0
        else:
            score = area

        if score > max_score:
            max_score = score
            best_label = i

    # 4. 重绘 Mask
    new_alpha = np.zeros_like(alpha)
    if best_label != -1:
        new_alpha[labels == best_label] = 255
    else:
        # 如果所有候选者都被过滤了（比如都在边缘），说明这张图处理失败，输出全黑
        return np.zeros_like(rgba_image)

    # 5. 更新 RGBA 图片
    # RGB 通道保持不变，只替换清洗后的 Alpha 通道
    cleaned_rgba = rgba_image.copy()
    cleaned_rgba[:, :, 3] = new_alpha

    return cleaned_rgba


def process_single_image(img_path, session):
    original_img = cv2.imread(img_path)
    if original_img is None: return None
    h, w = original_img.shape[:2]

    # --- Step 1: 物理裁剪 ---
    margin_h = int(h * CROP_RATIO)
    margin_w = int(w * CROP_RATIO)
    crop_img = original_img[margin_h:h - margin_h, margin_w:w - margin_w]
    if crop_img.size == 0: return None

    # --- Step 2: 提亮增强 ---
    input_crop = apply_clahe(crop_img)

    # --- Step 3: AI 推理 ---
    _, enc_img = cv2.imencode(".jpg", input_crop)
    output_bytes = remove(enc_img.tobytes(), session=session)
    nparr = np.frombuffer(output_bytes, np.uint8)
    crop_result_rgba = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # --- Step 4: 原位贴回 ---
    crop_alpha = crop_result_rgba[:, :, 3]
    final_alpha = np.zeros((h, w), dtype=np.uint8)
    final_alpha[margin_h:h - margin_h, margin_w:w - margin_w] = crop_alpha

    b, g, r = cv2.split(original_img)
    rough_rgba = cv2.merge([b, g, r, final_alpha])

    # --- Step 5: 【新增】二次清洗 ---
    # 这里不需要再跑 AI，直接用算法筛选
    final_clean_rgba = post_process_cleanup(rough_rgba, w, h)

    return final_clean_rgba


def main():
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print("✅ GPU 加速已启用")
    else:
        print("⚠️ 使用 CPU 运行")

    print(f"加载模型: {MODEL_NAME}...")
    session = new_session(model_name=MODEL_NAME, providers=providers)

    all_images = glob.glob(os.path.join(INPUT_ROOT, "**", "*.jpg"), recursive=True)
    print(f"开始处理 {len(all_images)} 张图片 (裁剪-提亮-清洗)...")

    for img_path in tqdm(all_images):
        try:
            result_rgba = process_single_image(img_path, session)

            if result_rgba is not None:
                relative_path = os.path.relpath(img_path, INPUT_ROOT)
                save_path = os.path.join(OUTPUT_ROOT, relative_path)
                save_path = save_path.replace(".jpg", ".png")

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, result_rgba)

        except Exception as e:
            print(f"Error: {img_path} - {e}")


if __name__ == "__main__":
    main()