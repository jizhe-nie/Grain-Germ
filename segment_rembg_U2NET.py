import os
import glob
from rembg import remove, new_session
from PIL import Image
import io
from tqdm import tqdm

# ================= 配置 =================
INPUT_ROOT = "output/results_seeds"  # 上一步裁剪好的图片目录
OUTPUT_ROOT = "output/segmented_seeds_ai"  # 输出目录


# =======================================

def main():
    # 初始化模型 session (第一次运行会自动下载约 170MB 的 u2net 模型)
    # 如果是针对边缘细节（如根须），可以使用 'isnet-general-use' 模型，效果可能更好
    session = new_session(model_name='u2net')

    # 获取所有图片列表
    all_images = glob.glob(os.path.join(INPUT_ROOT, "**", "*.jpg"), recursive=True)

    print(f"检测到 {len(all_images)} 张图片，开始 AI 分割...")

    for img_path in tqdm(all_images):
        try:
            # 1. 构造输出路径
            # 保持原始的文件夹结构: output/seed_xx/001.png
            relative_path = os.path.relpath(img_path, INPUT_ROOT)
            save_path = os.path.join(OUTPUT_ROOT, relative_path)
            save_path = save_path.replace(".jpg", ".png")  # 转为PNG以保存透明通道

            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # 2. 读取与处理
            with open(img_path, 'rb') as i:
                input_data = i.read()

            # 这里执行核心分割
            output_data = remove(input_data, session=session)

            # 3. 保存
            with open(save_path, 'wb') as o:
                o.write(output_data)

        except Exception as e:
            print(f"处理出错: {img_path}, 错误: {e}")


if __name__ == "__main__":
    main()