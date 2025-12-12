import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import cv2
import numpy as np
import os
import math
import albumentations as A  # 强大的增强库
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ================= 配置区域 =================
DATA_ROOT = "../dataset/stage2_data"
IMG_DIR = os.path.join(DATA_ROOT, "images")
LABEL_FILE = os.path.join(DATA_ROOT, "keypoint_labels.txt")

BATCH_SIZE = 32  # 增加 Batch Size
LR = 1e-4  # 学习率
EPOCHS = 150  # 增加轮数，因为有数据增强，模型需要看更多次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = 112


# ===========================================

# --- 1. 定义人脸关键点常用的 Wing Loss ---
# 相比 MSE，它对小误差更敏感，能显著提升定位精度
class WingLoss(nn.Module):
    def __init__(self, w=10.0, epsilon=2.0):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * math.log(1 + w / epsilon)

    def forward(self, pred, target):
        diff = pred - target
        abs_diff = torch.abs(diff)
        flag = (abs_diff < self.w).float()
        loss = flag * (self.w * torch.log(1 + abs_diff / self.epsilon)) + \
               (1 - flag) * (abs_diff - self.C)
        return loss.mean()


# --- 2. 定义增强型数据集 ---
class AugmentedRiceDataset(Dataset):
    def __init__(self, label_file, img_dir, is_train=True):
        self.img_dir = img_dir
        self.samples = []
        self.is_train = is_train

        with open(label_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 3:
                    self.samples.append((parts[0], float(parts[1]), float(parts[2])))

        # 定义训练时的数据增强策略
        # 注意：这里我们同时处理 4 通道数据稍微麻烦，
        # 所以策略是：先读 RGB 做几何变换，Lab 通道跟随变换
        self.transform = A.Compose([
            A.Rotate(limit=30, p=0.7),  # 随机旋转 +/- 30度
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.5),  # 平移和缩放
            A.RandomBrightnessContrast(p=0.5),  # 改变亮度对比度 (模拟光照变化)
            A.GaussNoise(p=0.2),  # 高斯噪声
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, x_norm, y_norm = self.samples[idx]
        img_path = os.path.join(self.img_dir, fname)

        # 读取 BGR
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return torch.zeros((4, 112, 112)), torch.tensor([0.0, 0.0])

        img_bgr = cv2.resize(img_bgr, (TARGET_SIZE, TARGET_SIZE))

        # 还原坐标到像素值 (增强库需要绝对坐标)
        kp_x = x_norm * TARGET_SIZE
        kp_y = y_norm * TARGET_SIZE

        # --- 数据增强核心逻辑 ---
        if self.is_train:
            # Albumentations 处理 RGB 图像
            augmented = self.transform(image=img_bgr, keypoints=[(kp_x, kp_y)])
            img_aug = augmented['image']
            kps_aug = augmented['keypoints']

            # 如果增强后点跑到图片外面去了，就回退到原始图片
            if len(kps_aug) == 0:
                img_final = img_bgr
                kp_x_final, kp_y_final = kp_x, kp_y
            else:
                img_final = img_aug
                kp_x_final, kp_y_final = kps_aug[0]

                # 限制坐标在 0-112 之间
                kp_x_final = min(max(0, kp_x_final), TARGET_SIZE)
                kp_y_final = min(max(0, kp_y_final), TARGET_SIZE)
        else:
            img_final = img_bgr
            kp_x_final, kp_y_final = kp_x, kp_y

        # --- 特征工程 (RGB + Lab-a) ---
        # 1. 提取 Lab-a
        img_lab = cv2.cvtColor(img_final, cv2.COLOR_BGR2Lab)
        _, a_channel, _ = cv2.split(img_lab)

        # 2. 转换 RGB
        img_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)

        # 3. 归一化 & 堆叠
        img_rgb = img_rgb.astype(np.float32) / 255.0
        a_channel = a_channel.astype(np.float32) / 255.0

        img_4c = np.dstack((img_rgb, a_channel))  # (112, 112, 4)
        img_tensor = torch.from_numpy(img_4c.transpose((2, 0, 1)))  # (4, 112, 112)

        # 4. 坐标重新归一化
        label_tensor = torch.tensor([kp_x_final / TARGET_SIZE, kp_y_final / TARGET_SIZE], dtype=torch.float32)

        return img_tensor, label_tensor


# --- 3. 更换模型: MobileNetV2 (人脸关键点常用骨干) ---
class MobileNetV2_4Ch(nn.Module):
    def __init__(self):
        super(MobileNetV2_4Ch, self).__init__()
        # 加载预训练权重
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # 修改第一层卷积以支持 4 通道
        # MobileNetV2 的第一层是 features[0][0]
        original_first_layer = self.model.features[0][0]
        new_first_layer = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)

        with torch.no_grad():
            new_first_layer.weight[:, :3, :, :] = original_first_layer.weight
            new_first_layer.weight[:, 3, :, :] = torch.mean(original_first_layer.weight, dim=1)

        self.model.features[0][0] = new_first_layer

        # 修改分类头为回归头
        # MobileNetV2 的分类头是 classifier[1]
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 2)  # 输出 x, y
        )

    def forward(self, x):
        return self.model(x)


def train():
    # 准备数据集 (训练集开启增强，验证集关闭)
    full_dataset = AugmentedRiceDataset(LABEL_FILE, IMG_DIR, is_train=True)
    val_check_dataset = AugmentedRiceDataset(LABEL_FILE, IMG_DIR, is_train=False)

    # 简单的划分索引
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    indices = list(range(total_len))
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    # 使用 Subset 创建数据集
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_check_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"训练集: {len(train_dataset)} (已开启实时增强), 验证集: {len(val_dataset)}")

    # 模型与优化器
    model = MobileNetV2_4Ch().to(DEVICE)

    # 使用 WingLoss
    criterion = WingLoss(w=10, epsilon=2)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)  # AdamW 防止过拟合

    # 学习率调整策略：余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float('inf')
    save_dir = "checkpoints_v2"
    os.makedirs(save_dir, exist_ok=True)

    print("开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                # 验证时也可以看 WingLoss，或者看 L1 距离
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)

        val_epoch_loss = val_loss / len(val_dataset)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}] Train Loss: {epoch_loss:.6f} | Val Loss: {val_epoch_loss:.6f}")

        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_mobile_kp_net.pth"))

    print(f"训练结束！最佳 Val Loss: {best_loss:.6f}")


if __name__ == "__main__":
    train()