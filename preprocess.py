import cv2
import numpy as np
import os
import re
import glob
import shutil

# ================= 配置区域 =================
INPUT_FOLDER = "dataset/80h"  # 输入图片所在的文件夹
OUTPUT_ROOT = "results_seeds"  # 输出结果的根目录
MAX_HOURS = 80  # 只处理前80小时
GRID_ROWS = 2  # 2行
GRID_COLS = 3  # 3列


# ===========================================

def parse_filename(filename):
    """
    解析文件名，例如: s508-10-090.jpg
    返回: hour (小时), global_idx (全局流水号)
    """
    # 正则匹配 s508-{小时}-{流水号}.jpg
    match = re.search(r's508-(\d+)-(\d+)', filename)
    if match:
        hour = int(match.group(1))
        idx = int(match.group(2))
        return hour, idx
    return None, None


def get_crop_boxes(img):
    """
    自动检测白色网格线来确定6个种子的切割坐标。
    使用投影法（Projection Profile）比轮廓检测更稳定。
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化，提取白色隔板
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 水平和垂直投影
    row_sum = np.sum(thresh, axis=1)  # 投影到Y轴
    col_sum = np.sum(thresh, axis=0)  # 投影到X轴

    # 寻找峰值（白色线条的位置）
    # 简单的逻辑：将图片分为 2x3，我们只需要找到大概的分割线
    # 为了防止噪点，我们只在预期范围内搜索

    # 寻找中间的水平分割线 (在图像高度的 1/3 到 2/3 之间搜索)
    mid_h_start, mid_h_end = int(h * 0.3), int(h * 0.7)
    h_split_idx = mid_h_start + np.argmax(row_sum[mid_h_start:mid_h_end])

    # 寻找垂直分割线 (我们需要2条垂直线，大概在 w/3 和 2w/3 处)
    w_one_third = int(w / 3)
    v_split_1 = int(w * 0.15) + np.argmax(col_sum[int(w * 0.15):int(w * 0.5)])  # 左边的一条
    v_split_2 = int(w * 0.5) + np.argmax(col_sum[int(w * 0.5):int(w * 0.85)])  # 右边的一条

    # 定义边界 (假设外边框也有白线，或者直接留一点边距)
    margin = 40  # 忽略边缘的像素宽度，防止切到盘子边框

    boxes = []
    # 坐标定义: y_starts, y_ends, x_starts, x_ends
    ys = [margin, h_split_idx, h - margin]
    xs = [margin, v_split_1, v_split_2, w - margin]

    # 生成 2x3 = 6 个坐标框 (y1, y2, x1, x2)
    # 顺序：先第一行(左中右)，后第二行(左中右)
    for r in range(2):
        for c in range(3):
            # 添加一点 padding 去除白线本身
            p = 15
            y1, y2 = ys[r] + p, ys[r + 1] - p
            x1, x2 = xs[c] + p, xs[c + 1] - p
            boxes.append((y1, y2, x1, x2))

    return boxes


def main():
    # 1. 准备输出目录
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)

    # 创建 54 个种子文件夹
    for i in range(1, 55):
        os.makedirs(os.path.join(OUTPUT_ROOT, f"seed_{i:02d}"))

    # 2. 扫描并整理文件列表
    # 结构: map[position_id][hour] = file_path
    print("正在扫描文件...")
    img_map = {i: {} for i in range(9)}  # 9个机位

    files = glob.glob(os.path.join(INPUT_FOLDER, "*.jpg"))
    for f in files:
        filename = os.path.basename(f)
        hour, idx = parse_filename(filename)

        if hour is not None and idx is not None:
            if hour < MAX_HOURS:  # 只取前80小时
                pos_id = idx % 9  # 关键逻辑：通过余数确定机位 [cite: 1, 33]
                img_map[pos_id][hour] = f

    # 3. 开始处理
    print(f"开始处理 9 个机位，共 {MAX_HOURS} 小时数据...")

    for pos_id in range(9):
        # 获取该机位的所有图片（按时间排序）
        hours = sorted(img_map[pos_id].keys())
        if not hours:
            continue

        print(f"正在处理机位 Position {pos_id} ...")

        # 3.1 计算该机位的切割坐标
        # 为了稳定，我们使用该机位的第一张图（Hour 0）来计算切割框
        # 并假设相机在滑轨上的重复定位精度足够高
        first_img_path = img_map[pos_id][hours[0]]
        base_img = cv2.imread(first_img_path)

        if base_img is None:
            print(f"无法读取图片: {first_img_path}")
            continue

        # 获得 6 个格子的坐标
        boxes = get_crop_boxes(base_img)

        # 3.2 遍历该机位的所有时间点
        for h in hours:
            img_path = img_map[pos_id][h]
            img = cv2.imread(img_path)
            if img is None: continue

            # 切割 6 颗种子
            for i, (y1, y2, x1, x2) in enumerate(boxes):
                # 种子全局编号计算
                # 机位0: 种子1-6; 机位1: 种子7-12 ...
                seed_global_id = (pos_id * 6) + (i + 1)

                # 切割 ROI
                roi = img[y1:y2, x1:x2]

                # 保存
                save_name = f"{h:03d}.jpg"  # 保存为 000.jpg, 001.jpg ...
                save_path = os.path.join(OUTPUT_ROOT, f"seed_{seed_global_id:02d}", save_name)
                cv2.imwrite(save_path, roi)

    print(f"处理完成！结果保存在 {OUTPUT_ROOT}")
    print("目录结构:")
    print(f"  {OUTPUT_ROOT}/seed_01/000.jpg (第1颗种子, 第0小时)")
    print(f"  {OUTPUT_ROOT}/seed_01/001.jpg (第1颗种子, 第1小时)")
    print("  ...")


if __name__ == "__main__":
    main()