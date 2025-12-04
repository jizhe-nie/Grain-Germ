import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from tqdm import tqdm

# ================= 配置区域 =================
# 数据路径
DATA_ROOT = "../dataset/stage2_data"
IMG_DIR = os.path.join(DATA_ROOT, "images")
LABEL_FILE = os.path.join(DATA_ROOT, "keypoint_labels.txt")

# 超参数
BATCH_SIZE = 16  # 样本少，Batch size 不宜太大
LR = 1e-4  # 学习率
EPOCHS = 100  # 训练轮数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================

class RiceBudDataset(Dataset):
    def __init__(self, label_file, img_dir):
        self.img_dir = img_dir
        self.samples = []

        # 读取标签文件
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 3:
                    fname, x, y = parts
                    self.samples.append((fname, float(x), float(y)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, x, y = self.samples[idx]
        img_path = os.path.join(self.img_dir, fname)

        # 1. 读取图片 (BGR)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            # 容错处理：如果读不到图，返回全0 tensor
            return torch.zeros((4, 112, 112)), torch.tensor([0.0, 0.0])

        img_bgr = cv2.resize(img_bgr, (112, 112))

        # 2. 核心特征工程：提取 Lab-'a' 通道
        # 你的导师文档重点强调：'a'通道能最大化区分芽尖和谷壳
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        _, a_channel, _ = cv2.split(img_lab)

        # 3. 转换 RGB 用于常规特征提取
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 4. 归一化 (0~1)
        img_rgb = img_rgb.astype(np.float32) / 255.0
        a_channel = a_channel.astype(np.float32) / 255.0

        # 5. 通道堆叠 (H, W, 3) + (H, W) -> (H, W, 4)
        # 最终形状需转换为 PyTorch 格式 (C, H, W) -> (4, 112, 112)
        img_4c = np.dstack((img_rgb, a_channel))
        img_tensor = torch.from_numpy(img_4c.transpose((2, 0, 1)))

        # 6. 标签转 Tensor
        label_tensor = torch.tensor([x, y], dtype=torch.float32)

        return img_tensor, label_tensor


class BudKPNet(nn.Module):
    def __init__(self):
        super(BudKPNet, self).__init__()
        # 加载 ResNet18 预训练模型
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # --- 关键修改：第一层卷积 (支持 4 通道) ---
        # 原始是 (64, 3, 7, 7)，我们改成 (64, 4, 7, 7)
        original_weights = self.backbone.conv1.weight.data
        new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 权重初始化策略：前3通道复制 ImageNet 权重，第4通道取 RGB 均值
        # 这样能保留预训练的纹理特征，同时快速适应新通道
        with torch.no_grad():
            new_conv1.weight[:, :3, :, :] = original_weights
            new_conv1.weight[:, 3, :, :] = torch.mean(original_weights, dim=1)

        self.backbone.conv1 = new_conv1

        # --- 关键修改：最后一层全连接 (输出 2 个坐标) ---
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.backbone(x)


def train():
    # 1. 准备数据
    full_dataset = RiceBudDataset(LABEL_FILE, IMG_DIR)

    # 划分训练集(80%) 和 验证集(20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")

    # 2. 初始化模型
    model = BudKPNet().to(DEVICE)
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_loss = float('inf')
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # 3. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # 训练阶段
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)

        val_epoch_loss = val_loss / len(val_dataset)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] Train Loss: {epoch_loss:.6f} | Val Loss: {val_epoch_loss:.6f}")

        # 保存最佳模型
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_bud_kp_net.pth"))
            # print("  >>> 最佳模型已保存")

    print(f"训练结束！最佳验证集 Loss: {best_loss:.6f}")
    print(f"模型已保存至: {os.path.join(save_dir, 'best_bud_kp_net.pth')}")


if __name__ == "__main__":
    # Windows 下多进程读取数据需要在 main 块中运行
    train()