import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from dataset import BridgeDataset, custom_collate_fn
from train import Trainable_VLM

def calculate_iou(pred_mask, gt_mask):
    """计算单个样本的 IoU (交并比)"""
    pred = pred_mask.view(-1).bool()
    gt = gt_mask.view(-1).bool()
    intersection = (pred & gt).sum().float()
    union = (pred | gt).sum().float()
    if union == 0: return 1.0 # 如果都没有病害，且预测也没有，IoU为1
    return (intersection / union).item()

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("⏳ 正在加载测试数据和模型...")
    
    # 假设你已经切分出了一个 test.json (如果没有，就先用训练集跑个分看看)
    test_dataset = BridgeDataset('data/datasets/metadata_ready.json')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    model = Trainable_VLM().to(device)
    model.load_state_dict(torch.load("bridge_vlm_epoch_5.pth", map_location=device))
    model.eval() # 开启评估模式
    
    total_iou = 0.0
    all_pred_types =[]
    all_gt_types =[]
    
    print("🚀 开始全量评估计算...")
    with torch.no_grad():
        for step, (b_global, b_patches, b_text, b_mask, b_type, b_grade) in enumerate(test_loader):
            b_mask = b_mask.to(device)
            b_type = b_type.cpu().numpy()[0] # 真实的8维多标签向量
            
            # 模型预测
            heatmap, mask, pred_types, pred_grades = model(b_global, b_patches, [b_text], device)
            
            # 1. 计算 mIoU
            iou = calculate_iou(mask, b_mask)
            total_iou += iou
            
            # 2. 收集分类结果用于计算 F1
            pred_types_sigmoid = torch.sigmoid(pred_types[0]).cpu().numpy()
            pred_binary = (pred_types_sigmoid > 0.5).astype(int) # 大于0.5视为预测有该病害
            
            all_pred_types.append(pred_binary)
            all_gt_types.append(b_type.astype(int))
            
            if step % 100 == 0:
                print(f"已评估 {step}/{len(test_loader)} 张图片...")

    # --- 最终指标计算 ---
    # 平均 IoU
    mIoU = total_iou / len(test_loader)
    
    # 微平均 F1-Score (Micro F1，适用于多标签分类)
    f1 = f1_score(all_gt_types, all_pred_types, average='micro')
    
    print("\n" + "="*50)
    print("🏆 论文/报告 指标计算完成！可以填入表格了：")
    print(f"👉 mIoU (定位精度) : {mIoU * 100:.2f} %")
    print(f"👉 F1-Score (综合指标): {f1 * 100:.2f} %")
    print("="*50)

if __name__ == "__main__":
    evaluate_model()