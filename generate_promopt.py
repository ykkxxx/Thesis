import json

# ================= 1. 配置路径 =================
INPUT_JSON = 'data/datasets/metadata.json'
OUTPUT_JSON = 'data/datasets/metadata_ready.json' # 这是最终喂给大模型的数据

# ================= 2. 中英文病害映射字典 =================
# 把你的 8 个中文类别精准翻译成工程英文
disease_dict = {
    '腐蚀': 'Corrosion',
    '裂缝': 'Crack',
    '退化混凝土': 'Spalling',
    '混凝土空洞': 'Concrete Void',
    '潮湿': 'Moisture',
    '路面劣化': 'Pavement Deterioration',
    '收缩裂缝': 'Shrinkage Crack',
    '底层收缩裂缝': 'Base Shrinkage Crack'
}

# ================= 3. 开始生成提示词 =================
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"正在处理 {len(data)} 张图片的提示词...")

for item in data:
    diseases = item['diseases']
    detected_types = set() # 用 set 去重，比如一张图有3个腐蚀，句子中只提一次
    reports =[]
    
    for d in diseases:
        en_type = disease_dict.get(d['type'], 'Unknown Disease')
        ratio = d['size_ratio']
        
        # 【巧用数据】：根据病害面积大小，自动评定等级 (Grade)
        # 如果病害面积超过 5%，就是严重(Level II)，否则是轻微(Level I)
        if ratio > 0.05:
            grade = "Level II (Severe)"
        else:
            grade = "Level I (Minor)"
            
        detected_types.add(en_type)
        
        # 模仿你的【图3】，生成结构化输出报告
        report = f"Detected {en_type} on structure. Grade: {grade}."
        reports.append(report)
        
    # 模仿你的【图1】，生成给 CLIP Text Encoder 的输入提示词
    types_str = " and ".join(list(detected_types))
    clip_prompt = f"A photo of a bridge structure showing {types_str} disease."
    
    # 将生成的句子写回字典里
    item['clip_prompt'] = clip_prompt
    item['target_reports'] = reports

# ================= 4. 保存最终版本 =================
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"✅ 提示词生成完毕！最终数据已保存至: {OUTPUT_JSON}")