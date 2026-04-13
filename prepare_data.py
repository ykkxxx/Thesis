import os
import cv2
import numpy as np
import yaml
import json

# ================= 1. 配置你的专属路径 =================
YAML_PATH = 'data/data.yaml'

IMG_DIR = 'data/datasets/images'

LABEL_DIR = 'data/datasets/labels'

# 以下是我们即将生成的 3 个新数据夹，统一放在 datasets 目录下方便管理
PATCH_DIR = 'data/datasets/patches'  # 存放局部病害切图（给 Diff Attention 用）
MASK_DIR = 'data/datasets/masks'  # 存放 0/1 掩码白块图（给图 3 的 Loss 用）
META_FILE = 'data/datasets/metadata.json'  # 存放文本元数据

# 自动创建输出文件夹
os.makedirs(PATCH_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

# ================= 2. 解析 yaml 获取中文病害名 =================
# 注意：一定要加 encoding='utf-8'，因为你的标签是中文的！
with open(YAML_PATH, 'r', encoding='utf-8') as f:
    data_cfg = yaml.safe_load(f)
class_names = data_cfg.get('names', [])
print(f"✅ 成功读取到 {len(class_names)} 个类别: {class_names}")

metadata_list = []

# ================= 3. 开始遍历处理 =================
print("⏳ 正在拼命切图和画Mask中，请稍候...")
for img_name in os.listdir(IMG_DIR):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    # 把 .png 或 .jpg 替换为 .txt 找到对应的标签名
    label_name = os.path.splitext(img_name)[0] + '.txt'
    label_path = os.path.join(LABEL_DIR, label_name)

    # 读取原图
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 无法读取图片: {img_path}")
        continue
    h, w, _ = img.shape

    # 创建一个和原图一样大的纯黑背景
    mask = np.zeros((h, w), dtype=np.uint8)
    disease_info_list = []

    # 检查这张图片有没有对应的 txt 标注
    if os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5: continue

                class_id = int(parts[0])
                c_x, c_y, box_w, box_h = map(float, parts[1:5])

                # 将 0~1 的比例还原成真实的像素坐标
                real_w, real_h = int(box_w * w), int(box_h * h)
                x_top_left = int((c_x * w) - (real_w / 2))
                y_top_left = int((c_y * h) - (real_h / 2))
                x_bottom_right = x_top_left + real_w
                y_bottom_right = y_top_left + real_h

                # 防止坐标跑到图片外面去
                x_top_left, y_top_left = max(0, x_top_left), max(0, y_top_left)
                x_bottom_right, y_bottom_right = min(w, x_bottom_right), min(h, y_bottom_right)

                # 获取中文名字（比如 '潮湿', '裂缝'）
                disease_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

                # --- 核心任务 A: 切割局部病害小图 ---
                patch = img[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
                if patch.size > 0:
                    patch_name = f"{os.path.splitext(img_name)[0]}_{disease_name}_{idx}.jpg"
                    cv2.imwrite(os.path.join(PATCH_DIR, patch_name), patch)

                # --- 核心任务 B: 在纯黑背景上画 255 的纯白块 ---
                cv2.rectangle(mask, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), 255, -1)

                disease_info_list.append({
                    "type": disease_name,
                    "location_box": [x_top_left, y_top_left, x_bottom_right, y_bottom_right],
                    "size_ratio": box_w * box_h
                })

    # 保存那张画满白块的 Mask 掩码图
    mask_name = os.path.splitext(img_name)[0] + '_mask.png'
    cv2.imwrite(os.path.join(MASK_DIR, mask_name), mask)

    # --- 核心任务 C: 记录 JSON 字典 ---
    if disease_info_list:
        metadata_list.append({
            "image_id": img_name,
            "global_view_path": img_path,
            "mask_path": os.path.join(MASK_DIR, mask_name),
            "diseases": disease_info_list
        })

# 把所有信息打包存成 JSON
with open(META_FILE, 'w', encoding='utf-8') as f:
    json.dump(metadata_list, f, indent=4, ensure_ascii=False)

print("\n 全部处理完成！快去文件夹里看看你的战利品吧！")