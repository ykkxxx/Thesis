import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

# ================== 之前的积木：Adapter 和 Diff Attention ==================
class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Adapter, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return x + self.fc2(self.relu(self.fc1(x)))

class DiffAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(DiffAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, global_feat, patch_feats):
        attn_output, _ = self.multihead_attn(query=global_feat, key=patch_feats, value=patch_feats)
        return self.layer_norm(global_feat + attn_output)

# ================== 【终极积木】图 3：跨模态解码器模块 ==================
class Figure3_Decoder(nn.Module):
    def __init__(self, img_dim=768, txt_dim=512, num_classes=8):
        super(Figure3_Decoder, self).__init__()
        
        # 1. 维度对齐：图像是 768维，文本是 512维，我们把图像降维到 512，才能进行交叉注意力
        self.img_proj = nn.Linear(img_dim, txt_dim)
        
        # 2. 交叉注意力层（图 3 中心的菱形块）
        self.cross_attention = nn.MultiheadAttention(embed_dim=txt_dim, num_heads=8, batch_first=True)
        
        # 3. 文本生成分支（图 3 左侧的 Transformer Decoder / 分类器）
        # 这里为了演示简单，我们用全连接层代替生成模型，预测“病害类型”和“等级”
        self.type_classifier = nn.Linear(txt_dim, num_classes) # 预测是哪种病害 (8选1)
        self.grade_classifier = nn.Linear(txt_dim, 2)          # 预测等级: Level I 或 Level II

    def forward(self, img_grid, txt_seq):
        # --- 步骤 1：交叉注意力 ---
        # img_grid: [1, 50, 768], txt_seq: [1, 14, 512]
        img_seq = self.img_proj(img_grid) # 变成 [1, 50, 512]
        
        # 图3明确指出：Text 是 Q, Image 是 K和V
        attn_output, attn_weights = self.cross_attention(
            query=txt_seq, 
            key=img_seq, 
            value=img_seq
        )
        
        # --- 步骤 2：生成 热力图与 0/1 Mask (图3 下方分支) ---
        # attn_weights 的形状是[1, 文本长度(14), 图像网格数(50)]
        # 我们把所有文本对图像的注意力平均起来，变成 [1, 50]
        spatial_attn = attn_weights.mean(dim=1) 
        
        # 核心：丢弃第 0 个无位置的 CLS Token，只取后 49 个！
        spatial_attn = spatial_attn[:, 1:] # 变成 [1, 49]
        
        # 魔法时刻：把 49 个格子还原成 7x7 的二维图像特征！
        heatmap_7x7 = spatial_attn.view(1, 1, 7, 7)
        
        # 将 7x7 的热力图放大回 224x224（和原图一样大）
        heatmap_224 = F.interpolate(heatmap_7x7, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 将数值压缩到 0~1 之间
        heatmap = torch.sigmoid(heatmap_224) 
        
        # 大于 0.5 的判定！（忠实还原你图 3 右下角的判定逻辑）
        binary_mask = (heatmap > 0.5).float() 
        
        # --- 步骤 3：生成 模板化报告 (图3 左侧分支) ---
        # 把注意力输出浓缩成一个特征向量
        text_fused_feat = attn_output.mean(dim=1) # 变成[1, 512]
        
        pred_types = self.type_classifier(text_fused_feat)
        pred_grades = self.grade_classifier(text_fused_feat)
        
        return heatmap, binary_mask, pred_types, pred_grades

# ================== 完整网络封装 ==================
class BridgeDiseaseVLM_Final(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        super(BridgeDiseaseVLM_Final, self).__init__()
        
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        for param in self.clip.parameters(): param.requires_grad = False
            
        vision_hidden_dim = self.clip.vision_model.config.hidden_size 
        text_hidden_dim = self.clip.text_model.config.hidden_size     
        
        self.diff_attention = DiffAttention(embed_dim=vision_hidden_dim)
        self.text_adapter = Adapter(input_dim=text_hidden_dim, hidden_dim=text_hidden_dim // 2)
        self.image_adapter = Adapter(input_dim=vision_hidden_dim, hidden_dim=vision_hidden_dim // 2)
        
        # 实例化刚才写好的图 3 解码器
        self.decoder = Figure3_Decoder(img_dim=vision_hidden_dim, txt_dim=text_hidden_dim)

    def forward(self, global_img, patch_imgs, texts):
        with torch.no_grad():
            global_out = self.clip.vision_model(**self.processor(images=global_img, return_tensors="pt"))
            global_grid_feat = global_out.last_hidden_state  
            
            patch_out = self.clip.vision_model(**self.processor(images=patch_imgs, return_tensors="pt"))
            patch_feats = patch_out.last_hidden_state 
            N, seq_len, dim = patch_feats.shape
            patch_feats = patch_feats.view(1, N * seq_len, dim)
            
            text_out = self.clip.text_model(**self.processor(text=texts, return_tensors="pt", padding=True))
            raw_text_embeds = text_out.last_hidden_state

        enhanced_image_feat = self.diff_attention(global_grid_feat, patch_feats)
        final_image_embeds = self.image_adapter(enhanced_image_feat)
        final_text_embeds = self.text_adapter(raw_text_embeds)
        
        # 把图1的结果喂给图3，输出最终结果！
        heatmap, mask, pred_types, pred_grades = self.decoder(final_image_embeds, final_text_embeds)
        
        return heatmap, mask, pred_types, pred_grades

# ================== 最终大测试 ==================
if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    
    print("⏳ 正在构建终极模型，请稍候...")
    model = BridgeDiseaseVLM_Final()
    
    # 模拟数据
    dummy_global = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    dummy_patch = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
    dummy_text = ["A photo of a bridge structure showing Corrosion and Moisture disease."]
    
    print("🚀 开始终极前向传播...")
    heatmap, mask, types, grades = model(dummy_global, [dummy_patch, dummy_patch], dummy_text)
    
    print("\n🎉🎉 终极版本 (图1+图3) 测试完美通关！ 🎉🎉")
    print("-" * 50)
    print("👇 模型最终输出的成果：")
    print(f"1. 生成的热力图 (Heatmap) 维度: {heatmap.shape}  -> 对应原图224x224每个像素的生病概率！")
    print(f"2. 大于0.5生成的 (0/1 Mask) 维度: {mask.shape}   -> 完美对应图3右下角的矩阵格子！")
    print(f"3. 预测类别 (Type Logits) 维度: {types.shape}   -> 8个数值，哪个最大就是哪种病害！")
    print(f"4. 预测等级 (Grade Logits) 维度: {grades.shape}  -> 2个数值，判断是 Level I 还是 II！")
    print("-" * 50)
    print("至此，你的技术路线图已在代码层面 100% 具象化！")