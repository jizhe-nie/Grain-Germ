import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from ultralytics import YOLO
import os
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 模型路径 (请确保路径正确)
YOLO_MODEL_PATH = "runs/detect/rice_seed_roi_22/weights/best.pt"
KP_MODEL_PATH = "checkpoints/best_bud_kp_net.pth"

# 2. 输入文件夹 (你指定的大图文件夹)
TEST_IMAGE_DIR = "../dataset/germination_nonuse"

# 3. 输出文件夹 (自动创建，用于保存结果)
OUTPUT_DIR = "../dataset/germination_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 4. 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================

# --- 定义 Bud-KP-Net 网络结构 (必须与训练时一致) ---
class BudKPNet(nn.Module):
    def __init__(self):
        super(BudKPNet, self).__init__()
        # 推理时不需要下载预训练权重 (weights=None)
        self.backbone = models.resnet18(weights=None)

        # 修改第一层卷积为 4 通道
        original_weights = self.backbone.conv1.weight.data
        new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.conv1 = new_conv1

        # 修改全连接层输出 2 个坐标
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.backbone(x)


def load_models():
    print(f"正在加载模型至 {DEVICE}...")
    # 1. 加载 YOLO
    yolo_model = YOLO(YOLO_MODEL_PATH)

    # 2. 加载 Bud-KP-Net
    kp_model = BudKPNet().to(DEVICE)
    kp_model.load_state_dict(torch.load(KP_MODEL_PATH, map_location=DEVICE))
    kp_model.eval()

    return yolo_model, kp_model


def preprocess_stage2(img_crop):
    """
    将裁剪下来的种子图处理成 (1, 4, 112, 112) 的 Tensor
    [cite_start]包含核心步骤：提取 Lab-'a' 通道并堆叠 [cite: 38]
    """
    # 1. Resize 到标准尺寸
    img_resized = cv2.resize(img_crop, (112, 112))

    # 2. 提取 Lab-a 通道
    img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)
    _, a_channel, _ = cv2.split(img_lab)

    # 3. 准备 RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # 4. 归一化 (0-1)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    a_channel = a_channel.astype(np.float32) / 255.0

    # 5. 堆叠 & 维度变换
    img_4c = np.dstack((img_rgb, a_channel))  # (112, 112, 4)
    img_tensor = torch.from_numpy(img_4c.transpose((2, 0, 1)))  # (4, 112, 112)
    img_tensor = img_tensor.unsqueeze(0)  # (1, 4, 112, 112) batch维度

    return img_tensor.to(DEVICE)


def run_batch_inference():
    yolo_model, kp_model = load_models()

    # 获取文件夹内所有图片
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"错误: 在 {TEST_IMAGE_DIR} 中没有找到图片。")
        return

    print(f"开始批量处理 {len(image_files)} 张图片...")

    for img_name in tqdm(image_files):
        img_path = os.path.join(TEST_IMAGE_DIR, img_name)

        # 读取原始大图
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"无法读取: {img_name}")
            continue

        frame_vis = frame.copy()

        # --- 阶段一: YOLO 粗定位 ---
        # conf=0.4: 置信度阈值，如果漏检可以调低，误检可以调高
        results = yolo_model(frame, conf=0.4, verbose=False)

        if len(results[0].boxes) == 0:
            # 如果没检测到种子，直接保存原图
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), frame_vis)
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # 边界检查，防止越界崩溃
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1: continue

            # 绘制种子框 (绿色)
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 裁剪 ROI
            crop = frame[y1:y2, x1:x2]

            # --- 阶段二: 关键点回归 ---
            input_tensor = preprocess_stage2(crop)

            with torch.no_grad():
                # 预测 normalized 坐标 (0~1)
                pred = kp_model(input_tensor)
                px, py = pred[0].cpu().numpy()

            # 映射回原图坐标
            roi_w = x2 - x1
            roi_h = y2 - y1

            real_x = int(x1 + px * roi_w)
            real_y = int(y1 + py * roi_h)

            # 绘制芽尖点 (红色实心圆)
            cv2.circle(frame_vis, (real_x, real_y), 5, (0, 0, 255), -1)

        # 保存结果图
        output_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(output_path, frame_vis)

    print(f"\n批量推理完成！")
    print(f"请查看结果文件夹: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    run_batch_inference()