import json

# 노트북 파일 읽기
notebook_path = 'detail_p2pnet.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 첫 번째 pred_probs를 pred_probs_bg로 변경
old_text1 = '''    "                # Filter out background class predictions\\n",
    "                pred_probs = F.softmax(pred_logits[b], dim=-1)\\n",
    "                # Get max prob excluding background (last class)\\n",
    "                foreground_probs = pred_probs[:, :-1]  # Exclude background\\n",'''

new_text1 = '''    "                # Filter out background class predictions\\n",
    "                pred_probs_bg = F.softmax(pred_logits[b], dim=-1)\\n",
    "                # Get max prob excluding background (last class)\\n",
    "                foreground_probs = pred_probs_bg[:, :-1]  # Exclude background\\n",'''

if old_text1 in content:
    content = content.replace(old_text1, new_text1)
    print("✅ 1st replacement: Changed pred_probs to pred_probs_bg (background filtering)")
else:
    print("❌ Old text not found for 1st replacement")

# 2. 두 번째 pred_probs를 pred_probs_match로 변경
old_text2 = '''    "                # Hungarian matching\\n",
    "                point_cost = torch.cdist(filtered_pred_points, gt_points, p=1)\\n",
    "                pred_probs = F.softmax(filtered_pred_logits, dim=-1)\\n",
    "                class_cost = -pred_probs[:, gt_classes]\\n",'''

new_text2 = '''    "                # Hungarian matching\\n",
    "                point_cost = torch.cdist(filtered_pred_points, gt_points, p=1)\\n",
    "                pred_probs_match = F.softmax(filtered_pred_logits, dim=-1)\\n",
    "                class_cost = -pred_probs_match[:, gt_classes]\\n",'''

if old_text2 in content:
    content = content.replace(old_text2, new_text2)
    print("✅ 2nd replacement: Changed pred_probs to pred_probs_match (Hungarian matching)")
else:
    print("❌ Old text not found for 2nd replacement")

# 파일 저장
with open(notebook_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\n✅ Notebook updated!")
print(f"✅ Removed duplicate pred_probs variables")
print(f"✅ Duplicate VALIDATION cell already deleted")
