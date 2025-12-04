#引入 Lab 颜色空间的 'a' 通道 。我们要验证这个通道是否真的能把“芽尖”突显出来。

import cv2
import numpy as np
import matplotlib.pyplot as plt


def verify_lab_channel(image_path):
    # 1. 读取原始图片 (BGR)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("图片读取失败")
        return

    # 2. 转换为 RGB (用于显示) 和 Lab
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)

    # 3. 提取 'a' 通道 (根据文档，这是核心区分特征 [cite: 36])
    # Lab 中：L=亮度, a=绿红轴, b=蓝黄轴
    l, a, b = cv2.split(img_lab)

    # 4. 可视化对比
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original RGB")

    plt.subplot(1, 2, 2)
    plt.imshow(a, cmap='gray')  # 用灰度图显示 'a' 通道
    plt.title("Lab 'a' Channel (Feature)")

    plt.show()
    print("请观察右图：芽尖部分是否比种子壳更黑（或更白，取决于正负值）？如果是，说明特征有效。")


# 换成你的一张测试图片路径
verify_lab_channel("../dataset/germination/262.jpg")

