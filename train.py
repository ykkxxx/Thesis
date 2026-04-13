import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

# 导入你前两天打下的江山
from dataset import BridgeDataset, custom_collate_fn
from model import BridgeDiseaseVLM_Final

# =====================================================================
# 1. 解决设备加载问题 (让模型适配显卡 GPU 训练)
# =====================================================================
class Trainable_VLM(BridgeDiseaseVLM_Final):
    """
    继承你写好的最终模型，稍微修改一下 forward 函数。
    目的：让 CLIP 的预处理器能把数据送到正确的显卡 GPU 上，防止报错。
    """
    def forward(self, global_img, patch_imgs, texts, device):
        with torch.no_grad():
            # 将处理器处理好的张量移动到 GPU
            global_inputs = self.processor(images=global_img, return_tensors="pt").to(device)
            global_out = self.clip.vision_model(**global_inputs)
            global_grid_feat = global_out.last_hidden_state  
            
            patch_inputs = self.processor(images=patch_imgs, return_tensors="pt").to(device)
            patch_out = self.clip.vision_model(**patch_inputs)
            patch_feats = patch_out.last_hidden_state 
            N, seq_len, dim = patch_feats.shape
            patch_feats = patch_feats.view(1, N * seq_len, dim)
            
            text_inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(device)
            text_out = self.clip.text_model(**text_inputs)
            raw_text_embeds = text_out.last_hidden_state

        # 后续网络流动保持不变
        enhanced_image_feat = self.diff_attention(global_grid_feat, patch_feats)
        final_image_embeds = self.image_adapter(enhanced_image_feat)
        final_text_embeds = self.text_adapter(raw_text_embeds)
        
        heatmap, mask, pred_types, pred_grades = self.decoder(final_image_embeds, final_text_embeds)
        return heatmap, mask, pred_types, pred_grades

# =====================================================================
# 2. 正式训练流程 (Training Loop)
# =====================================================================
def main():
    # 检测是否有显卡 (CUDA)，没有就用 CPU (会慢一点)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 当前使用的计算设备: {device}")

    # 1. 加载数据
    print("⏳ 正在加载数据集...")
    dataset = BridgeDataset('data/datasets/metadata_ready.json')
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    # 2. 加载模型
    print("🤖 正在初始化大模型 (并搬运到显卡上)...")
    model = Trainable_VLM().to(device)
    model.train() # 开启训练模式

    # 3. 定义你 图3 上的三个 Loss 尺子
    criterion_mask = nn.BCELoss()              # 用于评估 热力图生成得准不准
    criterion_type = nn.BCEWithLogitsLoss()    # 用于评估 病害种类猜得对不对
    criterion_grade = nn.CrossEntropyLoss()    # 用于评估 严重等级评判得准不准

    # 4. 定义优化器 (控制模型权重的更新步伐)
    # 我们只更新 requires_grad=True 的参数（即我们自定义的 Adapter 和 Attention）
    # CLIP 那些被冻结的参数不会参与更新，省下了极大的显存！
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable_params, lr=1e-4)

    # 5. 开始炼丹！(设置循环 5 轮)
    num_epochs = 5
    print("\n🚀 开始正式训练！")
    print("=" * 60)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for step, (b_global, b_patches, b_text, b_mask, b_type, b_grade) in enumerate(train_loader):
            # 将真实标签搬运到计算设备上
            b_mask = b_mask.to(device)
            b_type = b_type.to(device)
            b_grade = b_grade.to(device)

            # --- A. 前向传播 (让模型猜答案) ---
            # b_text 是个字符串，需要套个列表[] 送进去
            heatmap, mask, pred_types, pred_grades = model(b_global, b_patches, [b_text], device)

            # --- B. 计算误差 (对照标准答案算分) ---
            loss_mask = criterion_mask(heatmap, b_mask)
            loss_type = criterion_type(pred_types, b_type)
            loss_grade = criterion_grade(pred_grades, b_grade)
            
            # 总误差：图3路线图最终汇聚的点
            total_loss = loss_mask + loss_type + loss_grade

            # --- C. 反向传播 (模型自我反省并更新权重) ---
            optimizer.zero_grad() # 清空旧的记忆
            total_loss.backward() # 寻找误差来源
            optimizer.step()      # 更新 Adapter 里的参数

            epoch_loss += total_loss.item()

            # 每处理 50 张图，在屏幕上汇报一次成绩
            if step % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Step[{step}/{len(train_loader)}] "
                      f"Total Loss: {total_loss.item():.4f} "
                      f"(Mask: {loss_mask.item():.4f}, Type: {loss_type.item():.4f})")

        # 每一轮结束，打印这一轮的平均得分
        print("-" * 60)
        print(f"✅ Epoch {epoch+1} 完成! 平均 Loss: {epoch_loss/len(train_loader):.4f}")
        print("-" * 60)
        
        # 保存模型的学习成果 (权重文件)
        torch.save(model.state_dict(), f"bridge_vlm_epoch_{epoch+1}.pth")
        print(f"💾 模型权重已保存至 bridge_vlm_epoch_{epoch+1}.pth")

    print("🎉🎉🎉 恭喜！大模型训练圆满结束！ 🎉🎉🎉")

if __name__ == "__main__":
    main()