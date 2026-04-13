import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

# =====================================================================
# 1. 定义 PyTorch 数据集类 (Dataset)
# 作用：告诉 PyTorch 怎么去读取你那几千张图片和 json 文件
# =====================================================================
class BridgeDataset(Dataset):
    def __init__(self, json_path):
        # 1. 读取我们阶段二生成的“完美教材”
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # 2. 定义中文标签到数字 ID 的映射 (严格按照你 data.yaml 的顺序)
        self.class_names =['腐蚀', '裂缝', '退化混凝土', '混凝土空洞', 
                            '潮湿', '路面劣化', '收缩裂缝', '底层收缩裂缝']
        self.zh2id = {name: idx for idx, name in enumerate(self.class_names)}

        # 3. 对 Mask 掩码进行缩放处理 (因为模型最终输出的热力图是 224x224 的)
        self.mask_transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST), 
            T.ToTensor() # 自动将像素值转换到 0~1 之间
        ])

    def __len__(self):
        return len(self.data) # 返回总共有多少张图

    def __getitem__(self, idx):
        # 这个函数每次被调用，就会输出 "1张" 图片的所有配套数据
        item = self.data[idx]

        # --- A. 加载大图 (Global View) ---
        global_img = Image.open(item['global_view_path']).convert('RGB')

        # --- B. 动态裁剪局部图 (Patches) 和 提取标签 ---
        patches =[]
        # 注意：一张图可能有多种病害！所以我们用长度为 8 的向量，有病害的位置设为 1
        gt_type_vector = torch.zeros(8) 
        is_severe = 0 # 默认等级为 Level I (0)

        for d in item['diseases']:
            # 【高能技巧】直接从大图上根据坐标把小图裁剪下来，不用去读盘找小图了，速度极快！
            box = d['location_box']
            patch = global_img.crop((box[0], box[1], box[2], box[3]))
            
            # 【终极防爆修复】：防止抠出 1 像素高或宽的极度狭窄图片
            # 如果宽或高太小，CLIP处理器会宕机，我们把它稍微放大一点点
            patch_w, patch_h = patch.size
            if patch_w < 10 or patch_h < 10:
                patch = patch.resize((max(10, patch_w), max(10, patch_h)))
                
            patches.append(patch)

            # 标记病害类别 (Multi-hot)
            if d['type'] in self.zh2id:
                class_id = self.zh2id[d['type']]
                gt_type_vector[class_id] = 1.0
            
            # 标记病害等级 (只要有一个病害面积大于5%，整体就判定为严重 Level II)
            if d['size_ratio'] > 0.05:
                is_severe = 1

        # 如果没有检测到病害，给个默认切片防报错
        if len(patches) == 0:
            patches.append(global_img)

        # --- C. 加载 Mask (用作计算图3 Loss 的标准答案) ---
        mask_img = Image.open(item['mask_path']).convert('L')
        gt_mask = self.mask_transform(mask_img)
        gt_mask = (gt_mask > 0.5).float() # 确保只有纯粹的 0.0 和 1.0

        # --- D. 提取英文提示词 ---
        clip_prompt = item['clip_prompt']
        gt_grade = torch.tensor(is_severe, dtype=torch.long)

        # 返回装配好的数据包！
        return {
            "global_img": global_img,          # PIL Image (给 CLIP 用的)
            "patches": patches,                # List[PIL Image] (给 Diff Attention 用的)
            "text": clip_prompt,               # String 字符串
            "gt_mask": gt_mask,                # Tensor [1, 224, 224] (定位 Loss 标准答案)
            "gt_type": gt_type_vector,         # Tensor [8] (分类 Loss 标准答案)
            "gt_grade": gt_grade               # Tensor[] (等级 Loss 标准答案)
        }

# =====================================================================
# 2. 编写批量整理函数 (Collate Function)
# 因为 patches 的数量不固定（有的图2个病害，有的6个），默认的 DataLoader 会报错。
# 我们需要手写一个打包规则。
# =====================================================================
def custom_collate_fn(batch):
    # 【强制要求】：为了不修改你昨天的完美模型，我们规定每次只送 1 张大图进模型 (batch_size=1)
    item = batch[0]
    return (
        item['global_img'], 
        item['patches'], 
        item['text'], 
        item['gt_mask'].unsqueeze(0), # 增加 batch 维度变成 [1, 1, 224, 224]
        item['gt_type'].unsqueeze(0), # 增加 batch 维度变成 [1, 8]
        item['gt_grade'].unsqueeze(0) # 增加 batch 维度变成 [1]
    )

# =====================================================================
# 3. 本地跑一下测试，看数据传送带转不转！
# =====================================================================
if __name__ == "__main__":
    # 请核对你的 json 文件路径是否正确！
    JSON_PATH = 'data/datasets/metadata_ready.json' 
    
    print("⏳ 正在启动数据传送带...")
    my_dataset = BridgeDataset(JSON_PATH)
    
    # 实例化 DataLoader
    train_loader = DataLoader(
        my_dataset, 
        batch_size=1, # 强制为1，大模型微调的标配！
        shuffle=True, # 每次打乱顺序拿图片
        collate_fn=custom_collate_fn # 使用我们自定义的打包规则
    )
    
    print(f"数据集加载成功，总共有 {len(my_dataset)} 张训练数据！")
    
    # 我们试着从传送带上拿 1 个 Batch 的数据出来看看：
    for batch_idx, (b_global, b_patches, b_text, b_mask, b_type, b_grade) in enumerate(train_loader):
        print("-" * 50)
        print("🎉 成功从 DataLoader 抽出一条数据！来看看准备送给模型的是什么：")
        print(f"📝 英文提示词 (Prompt): {b_text}")
        print(f"🖼️ 全局大图格式: {type(b_global)}")
        print(f"🧩 局部切片数量: {len(b_patches)} 张")
        print(f"🎯 真实 Mask 答案维度: {b_mask.shape} (用于算 Loss_mask)")
        print(f"📊 真实 类别 答案维度: {b_type.shape} | 数据: {b_type}")
        print(f"📈 真实 等级 答案维度: {b_grade.shape} | 数据: {b_grade}")
        print("-" * 50)
        break # 测完第一条就停下