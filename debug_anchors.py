import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 앵커 생성 함수 (모델과 동일, numpy로 구현)
def generate_anchors(h_feat=16, w_feat=16, row=2, line=2):
    # Feature map cell centers
    y_centers = (np.arange(h_feat).astype(float) + 0.5) / h_feat
    x_centers = (np.arange(w_feat).astype(float) + 0.5) / w_feat
    
    xx, yy = np.meshgrid(x_centers, y_centers)  # numpy는 xy indexing이 기본
    
    # Sub-anchors
    sub_offsets = []
    for r in range(row):
        for l in range(line):
            dy = (r + 0.5) / row / h_feat - 0.5 / h_feat
            dx = (l + 0.5) / line / w_feat - 0.5 / w_feat
            sub_offsets.append((dx, dy))
    
    # Generate all anchors
    anchors = []
    for dx, dy in sub_offsets:
        anchor_x = xx + dx
        anchor_y = yy + dy
        anchor_pts = np.stack([anchor_x.flatten(), anchor_y.flatten()], axis=1)
        anchors.append(anchor_pts)
    
    anchors = np.vstack(anchors)
    return anchors

# 앵커 생성
anchors = generate_anchors(16, 16, 2, 2)
print(f"Total anchors: {len(anchors)}")
print(f"First 5 anchors (normalized [0,1]):")
for i in range(5):
    print(f"  {i}: x={anchors[i,0]:.4f}, y={anchors[i,1]:.4f}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# 왼쪽: 앵커만
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].set_aspect('equal')
axes[0].scatter(anchors[:, 0], anchors[:, 1], c='blue', s=10, alpha=0.5)
axes[0].set_title(f'Anchors (total: {len(anchors)})', fontsize=14, fontweight='bold')
axes[0].set_xlabel('X (width)')
axes[0].set_ylabel('Y (height)')
axes[0].invert_yaxis()  # 이미지 좌표계 (y축 반전)
axes[0].grid(True, alpha=0.3)

# 오른쪽: GT 샘플과 앵커 비교
import json
import glob

label_files = sorted(glob.glob('../../data/HnE_cell_detect/total_data/labels/*.json'))
label_file = label_files[5]  # 6번째 파일

with open(label_file) as f:
    data_json = json.load(f)

# GT 좌표 읽기 (현재 코드 방식)
gt_points = []
for coord in data_json["cordinates"]:
    if len(coord) < 5:
        continue
    y = coord[1]
    x = coord[2]
    h = coord[3]
    w = coord[4]
    if h > 50 or w > 50:
        continue
    
    # 픽셀 → 정규화
    center_x = (x + w//2) / 500.0
    center_y = (y + h//2) / 500.0
    gt_points.append([center_x, center_y])

gt_points = np.array(gt_points)

axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].set_aspect('equal')
axes[1].scatter(anchors[:, 0], anchors[:, 1], c='blue', s=5, alpha=0.3, label='Anchors')
axes[1].scatter(gt_points[:, 0], gt_points[:, 1], c='red', s=30, alpha=0.8, label='GT')
axes[1].set_title(f'Anchors vs GT ({len(gt_points)} cells)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('X (width)')
axes[1].set_ylabel('Y (height)')
axes[1].invert_yaxis()
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('anchor_debug.png', dpi=150)
print("\n✅ Saved: anchor_debug.png")
print(f"\n만약 GT(빨강)가 앵커(파랑)와 전혀 다른 위치에 있다면:")
print(f"  → 좌표 형식이 잘못되었거나 앵커 생성 코드가 잘못됨")
