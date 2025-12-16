import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from ultralytics import YOLO
import os
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 你的 YOLO 模型路径 (第一阶段)
YOLO_MODEL_PATH = "runs/detect/rice_seed_roi_2/weights/best.pt"

# 2. 你的新 MobileNet 关键点模型路径 (第二阶段)
KP_MODEL_PATH = "checkpoints_v2/best_mobile_kp_net.pth"

# 3. 输入文件夹 (未使用的图片)
TEST_IMAGE_DIR = "../dataset/germination_nonuse"

# 4. 输出文件夹 (自动创建)
OUTPUT_DIR = "../dataset/germination_results_advanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = 112


# ===========================================

# --- 定义 MobileNetV2 网络结构 (必须与训练代码完全一致) ---
class MobileNetV2_4Ch(nn.Module):
    def __init__(self):
        super(MobileNetV2_4Ch, self).__init__()
        # 推理时 weights=None
        self.model = models.mobilenet_v2(weights=None)

        # 修改第一层 (支持 4 通道)
        original_first_layer = self.model.features[0][0]
        new_first_layer = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.features[0][0] = new_first_layer

        # 修改输出层 (回归 2 个坐标)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        return self.model(x)


def load_models():
    print(f"正在加载模型至 {DEVICE}...")
    # 加载 YOLO
    yolo_model = YOLO(YOLO_MODEL_PATH)

    # 加载 MobileNetV2
    kp_model = MobileNetV2_4Ch().to(DEVICE)
    kp_model.load_state_dict(torch.load(KP_MODEL_PATH, map_location=DEVICE))
    kp_model.eval()

    return yolo_model, kp_model


def preprocess_stage2(img_crop):
    """
    预处理：Resize -> Lab特征提取 -> 归一化 -> 堆叠 -> Tensor
    """
    # 1. Resize
    img_resized = cv2.resize(img_crop, (TARGET_SIZE, TARGET_SIZE))

    # 2. 提取 Lab-a
    img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)
    _, a_channel, _ = cv2.split(img_lab)

    # 3. 准备 RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # 4. 归一化 (0-1)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    a_channel = a_channel.astype(np.float32) / 255.0

    # 5. 堆叠 (RGB + a)
    img_4c = np.dstack((img_rgb, a_channel))
    img_tensor = torch.from_numpy(img_4c.transpose((2, 0, 1)))
    img_tensor = img_tensor.unsqueeze(0)  # Batch dimension

    return img_tensor.to(DEVICE)


def run_batch_inference():
    yolo_model, kp_model = load_models()

    # 获取图片列表
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(valid_exts)]

    if not image_files:
        print(f"错误: 文件夹 {TEST_IMAGE_DIR} 是空的。")
        return

    print(f"开始处理 {len(image_files)} 张图片...")

    for img_name in tqdm(image_files):
        img_path = os.path.join(TEST_IMAGE_DIR, img_name)

        frame = cv2.imread(img_path)
        if frame is None: continue

        frame_vis = frame.copy()

        # --- 1. YOLO 粗定位 ---
        # conf=0.45: 稍微降低一点阈值，保证不漏检
        results = yolo_model(frame, conf=0.45, verbose=False)

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                # 越界修正
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1: continue

                # 画 YOLO 框 (绿色)
                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # --- 2. MobileNet 精定位 ---
                crop = frame[y1:y2, x1:x2]
                input_tensor = preprocess_stage2(crop)

                with torch.no_grad():
                    pred = kp_model(input_tensor)
                    px, py = pred[0].cpu().numpy()

                # 坐标映射
                roi_w = x2 - x1
                roi_h = y2 - y1
                real_x = int(x1 + px * roi_w)
                real_y = int(y1 + py * roi_h)

                # 画点 (青色实心 + 红色描边，高亮显示)
                cv2.circle(frame_vis, (real_x, real_y), 5, (255, 255, 0), -1)
                cv2.circle(frame_vis, (real_x, real_y), 7, (0, 0, 255), 1)

        # 保存图片
        cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), frame_vis)

    print(f"\n全部完成！结果保存在: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    run_batch_inference()