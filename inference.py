import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

# 导入你写好的模型
from train import Trainable_VLM

# =========================================================
# 1. 配置加载参数
# =========================================================
# 填入你跑完的最后一个权重的名字 (比如第5个epoch的权重)
WEIGHT_PATH = "bridge_vlm_epoch_5.pth" 
JSON_PATH = "data/datasets/metadata_ready.json"

# 你的病害字典
class_names =['腐蚀', '裂缝', '退化混凝土', '混凝土空洞', 
               '潮湿', '路面劣化', '收缩裂缝', '底层收缩裂缝']

# =========================================================
# 2. 加载训练好的大模型
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 使用设备: {device}")

print("🤖 正在唤醒大模型并注入训练好的记忆...")
model = Trainable_VLM().to(device)
# 加载你炼丹炼出来的权重
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
model.eval() # 开启考试（推理）模式，不再更新梯度！

# =========================================================
# 3. 抓取一张测试图片
# =========================================================
print("📸 正在抽取一张测试照片...")
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
    
# 我们随便抽第 10 张图片来做测试 (你可以随便改数字)
test_item = data[10] 

global_img_path = test_item['global_view_path']
global_img = Image.open(global_img_path).convert('RGB')
prompt_text = test_item['clip_prompt']

# 模拟切出小图 (推理时我们也需要给它切片)
patches = []
for d in test_item['diseases']:
    box = d['location_box']
    patch = global_img.crop((box[0], box[1], box[2], box[3]))
    if patch.size[0] < 10 or patch.size[1] < 10:
        patch = patch.resize((max(10, patch.size[0]), max(10, patch.size[1])))
    patches.append(patch)

if len(patches) == 0: patches.append(global_img)

# =========================================================
# 4. 让模型开始诊断！
# =========================================================
print(f"📥 输入提示词: {prompt_text}")
print("🧠 模型思考中...")

with torch.no_grad(): # 考试时不计算梯度
    # 模型输出：热力图, 0/1掩码, 类别Logit, 等级Logit
    heatmap, mask, pred_types, pred_grades = model(global_img, patches, [prompt_text], device)

# --- A. 解析模型给出的文字诊断报告 ---
# 取出概率大于0 (经过sigmoid大于0.5) 的病害类别
pred_types_sigmoid = torch.sigmoid(pred_types[0])
predicted_class_indices = (pred_types_sigmoid > 0.5).nonzero(as_tuple=True)[0]
detected_diseases = [class_names[i] for i in predicted_class_indices]

# 解析等级
predicted_grade = torch.argmax(pred_grades[0]).item()
grade_str = "Level II (严重)" if predicted_grade == 1 else "Level I (轻微)"

print("\n" + "="*50)
print("📝 【AI 诊断报告生成完毕】 (完美对应图3左侧的Template)")
if len(detected_diseases) > 0:
    for disease in detected_diseases:
        print(f"-> 发现病害 [Type]: {disease}")
    print(f"-> 整体评级[Grade]: {grade_str}")
else:
    print("-> 未检测到明显病害，结构健康！")
print("="*50)

# --- B. 绘制并保存酷炫的热力图 (完美对应图3中心) ---
print("🎨 正在生成病害定位热力图...")
# 把热力图从 GPU 拿回 CPU，变成 Numpy 矩阵
heatmap_np = heatmap.squeeze().cpu().numpy()

# 读取原图，准备把热力图叠加上去
img_cv = cv2.imread(global_img_path)
img_cv = cv2.resize(img_cv, (224, 224)) # 缩放到和热力图一样大

# 把 0~1 的热力图转换为 0~255 的伪彩色图 (Jet 颜色映射，红色代表高亮)
heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_np), cv2.COLORMAP_JET)

# 将原图和彩色热力图混合叠加 (0.5 透明度)
superimposed_img = cv2.addWeighted(img_cv, 0.5, heatmap_color, 0.5, 0)

# 保存最终的成果图
output_filename = "result_heatmap.jpg"
cv2.imwrite(output_filename, superimposed_img)

print(f"\n🎉 完美收官！请在文件夹中查看你的战果图片: {output_filename}")



## --- B. 绘制完美用于论文的对比图 (Side-by-Side) ---
print("🎨 正在生成用于论文的高级对比热力图...")

# 1. 拿回那张 224x224 的热力图矩阵
heatmap_np = heatmap.squeeze().cpu().numpy()

# 2. 读取最原始、最高清的图片
orig_img_cv = cv2.imread(global_img_path)
orig_h, orig_w = orig_img_cv.shape[:2]

# 3. 把 224x224 的热力图无损放大，拉伸回原图的真实分辨率！
heatmap_resized = cv2.resize(heatmap_np, (orig_w, orig_h))

# 4. 上色：把 0~1 的数字变成红黄蓝的伪彩色 (Jet)
heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
# 和原图融合叠加
superimposed_img = cv2.addWeighted(orig_img_cv, 0.6, heatmap_color, 0.4, 0)

# 5. 在【原图】上画出人工标注的真实框 (Ground Truth)，用作参考答案
img_with_gt_boxes = orig_img_cv.copy()
for d in test_item['diseases']:
    box = d['location_box']
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    # 画绿色粗框
    cv2.rectangle(img_with_gt_boxes, (x1, y1), (x2, y2), (0, 255, 0), 4)
    # 写上真实标签名字
    cv2.putText(img_with_gt_boxes, d['type'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# 6. 使用 Matplotlib 把两张图左右拼在一起
plt.figure(figsize=(16, 8))

# 左图：真实情况
plt.subplot(1, 2, 1)
# cv2读取是BGR，matplotlib画图需要RGB，所以转换一下颜色通道
plt.imshow(cv2.cvtColor(img_with_gt_boxes, cv2.COLOR_BGR2RGB))
plt.title("Ground Truth (Original with Boxes)", fontsize=16)
plt.axis('off')

# 右图：AI 预测的热力图
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title("AI Prediction Heatmap", fontsize=16)
plt.axis('off')

# 保存这张精美的对比图！
output_filename = "comparison_result.jpg"
plt.tight_layout()
plt.savefig(output_filename, dpi=300) # dpi=300 保证图片超高清，论文要求！
plt.close()

print(f"\n🎉 对比图已生成！请打开 {output_filename} 验收模型眼力！")








"""
# --- B. 绘制完美用于论文的对比图 (Side-by-Side) ---
print("🎨 正在生成用于论文的高级对比热力图...")

# 1. 拿回那张 224x224 的热力图矩阵
heatmap_np = heatmap.squeeze().cpu().numpy()

# 2. 读取最原始、最高清的图片
orig_img_cv = cv2.imread(global_img_path)
orig_h, orig_w = orig_img_cv.shape[:2]

# 3. 把 224x224 的热力图无损放大，拉伸回原图的真实分辨率！
heatmap_resized = cv2.resize(heatmap_np, (orig_w, orig_h))

# 4. 上色：把 0~1 的数字变成红黄蓝的伪彩色 (Jet)
heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
# 和原图融合叠加
superimposed_img = cv2.addWeighted(orig_img_cv, 0.6, heatmap_color, 0.4, 0)

# 5. 在【原图】上画出人工标注的真实框 (Ground Truth)，用作参考答案
img_with_gt_boxes = orig_img_cv.copy()
for d in test_item['diseases']:
    box = d['location_box']
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    # 画绿色粗框
    cv2.rectangle(img_with_gt_boxes, (x1, y1), (x2, y2), (0, 255, 0), 4)
    # 写上真实标签名字
    cv2.putText(img_with_gt_boxes, d['type'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# 6. 使用 Matplotlib 把两张图左右拼在一起
plt.figure(figsize=(16, 8))

# 左图：真实情况
plt.subplot(1, 2, 1)
# cv2读取是BGR，matplotlib画图需要RGB，所以转换一下颜色通道
plt.imshow(cv2.cvtColor(img_with_gt_boxes, cv2.COLOR_BGR2RGB))
plt.title("Ground Truth (Original with Boxes)", fontsize=16)
plt.axis('off')

# 右图：AI 预测的热力图
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title("AI Prediction Heatmap", fontsize=16)
plt.axis('off')

# 保存这张精美的对比图！
output_filename = "comparison_result.jpg"
plt.tight_layout()
plt.savefig(output_filename, dpi=300) # dpi=300 保证图片超高清，论文要求！
plt.close()

print(f"\n🎉 对比图已生成！请打开 {output_filename} 验收模型眼力！")
"""