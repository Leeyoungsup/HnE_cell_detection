"""
디버깅 스크립트: 예측이 한 점에 모이는 원인 진단

실행 방법:
노트북에서 Cell 9 (Training Setup)까지 실행 후,
아래 코드를 새 셀에 복사해서 실행하세요.
"""

print("\n" + "="*80)
print("🔍 DETR 예측 분석: 왜 한 점에 모이는가?")
print("="*80)

# 1️⃣ 타겟 라벨 확인
print("\n" + "="*80)
print("1️⃣ 타겟 라벨 확인 (Ground Truth가 제대로 들어가는지)")
print("="*80)

samples, targets = next(iter(train_loader))
samples = samples.to(device)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

for i, target in enumerate(targets[:2]):
    print(f"\n[Sample {i}]")
    print(f"  GT Points shape: {target['points'].shape}")
    print(f"  GT Labels shape: {target['labels'].shape}")
    print(f"  Total GT cells: {len(target['points'])}")
    print(f"  GT Points (처음 5개):")
    for j in range(min(5, len(target['points']))):
        pt = target['points'][j]
        label = target['labels'][j]
        print(f"    Cell {j}: pos=({pt[0]:.3f}, {pt[1]:.3f}), class={label} ({class_names[label.item()]})")
    print(f"  GT Points 통계:")
    print(f"    Mean: ({target['points'][:, 0].mean():.3f}, {target['points'][:, 1].mean():.3f})")
    print(f"    Std: ({target['points'][:, 0].std():.3f}, {target['points'][:, 1].std():.3f})")
    print(f"    Min: ({target['points'][:, 0].min():.3f}, {target['points'][:, 1].min():.3f})")
    print(f"    Max: ({target['points'][:, 0].max():.3f}, {target['points'][:, 1].max():.3f})")

print(f"\n✅ 타겟이 정상이라면: 다양한 위치 ([0, 1] 범위), 다양한 클래스")

# 2️⃣ 모델 예측 확인 (학습 전)
print("\n" + "="*80)
print("2️⃣ 모델 예측 확인 (초기화 직후 상태)")
print("="*80)

model.eval()
with torch.no_grad():
    outputs = model(samples)
    points_pred = outputs['pred_points'][0]  # 첫 번째 이미지
    logits_pred = outputs['pred_logits'][0]
    
    print(f"\n예측 shape:")
    print(f"  Points: {points_pred.shape}")  # [100, 2]
    print(f"  Logits: {logits_pred.shape}")  # [100, 7]
    
    print(f"\n예측 포인트 (처음 10개 query):")
    for i in range(10):
        print(f"  Query {i}: ({points_pred[i, 0]:.4f}, {points_pred[i, 1]:.4f})")
    
    print(f"\n예측 포인트 통계:")
    print(f"  Mean: ({points_pred[:, 0].mean():.4f}, {points_pred[:, 1].mean():.4f})")
    print(f"  Std: ({points_pred[:, 0].std():.4f}, {points_pred[:, 1].std():.4f})")
    print(f"  Min: ({points_pred[:, 0].min():.4f}, {points_pred[:, 1].min():.4f})")
    print(f"  Max: ({points_pred[:, 0].max():.4f}, {points_pred[:, 1].max():.4f})")
    
    unique_points = torch.unique(points_pred, dim=0)
    print(f"\n  Unique 예측 수: {len(unique_points)}/100")
    
    if points_pred.std() < 0.01:
        print(f"\n⚠️  경고: 모든 query가 거의 같은 점 예측! (std < 0.01)")
        print(f"\n원인: point_embed 마지막 layer가 0으로 초기화됨")
        print(f"  → MLP 출력 = 0")
        print(f"  → sigmoid(0) = 0.5")
        print(f"  → 모든 query가 (0.5, 0.5) 예측")
        print(f"\n✅ 이것은 정상입니다! (공식 DETR도 비슷한 초기화)")
        print(f"   학습이 시작되면 자연스럽게 분화됩니다.")
    else:
        print(f"\n✅ Query들이 분화되고 있습니다! (std = {points_pred.std():.4f})")

# 3️⃣ Loss 분석
print("\n" + "="*80)
print("3️⃣ Loss 상세 분석")
print("="*80)

model.train()
outputs = model(samples)
loss_dict = criterion(outputs, targets)
losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

print(f"\nLoss 값:")
print(f"  Total Loss: {losses.item():.4f}")
print(f"  CE Loss: {loss_dict['loss_ce'].item():.4f}")
print(f"  Point Loss: {loss_dict['loss_point'].item():.4f}")
print(f"  Cardinality Error: {loss_dict['cardinality_error'].item():.4f}")

# Hungarian matching 결과
with torch.no_grad():
    indices = criterion.matcher(outputs, targets)
    
    print(f"\nHungarian Matching 결과:")
    for i, (pred_idx, gt_idx) in enumerate(indices[:2]):
        print(f"\n  Sample {i}:")
        print(f"    Matched pairs: {len(pred_idx)}")
        print(f"    Matched query indices (처음 10개): {pred_idx[:10].tolist()}")
        print(f"    Matched GT indices (처음 10개): {gt_idx[:10].tolist()}")
        
        if len(pred_idx) > 0:
            matched_pred = outputs['pred_points'][i][pred_idx]
            matched_gt = targets[i]['points'][gt_idx]
            
            dist = torch.abs(matched_pred - matched_gt).mean(dim=1)
            print(f"    Point error (처음 5개): {[f'{d:.4f}' for d in dist[:5].tolist()]}")
            print(f"    Mean point error: {dist.mean():.4f}")

if loss_dict['loss_point'].item() > 0.3:
    print(f"\n⚠️  Point loss가 큽니다 (>0.3)")
    print(f"   초기 상태라면 정상: 모든 예측이 (0.5, 0.5)이므로 GT와 거리가 큼")
else:
    print(f"\n✅ Point loss 정상")

# 4️⃣ 학습 시뮬레이션
print("\n" + "="*80)
print("4️⃣ 학습 시뮬레이션 (10 steps)")
print("="*80)

initial_points = outputs['pred_points'][0, :10].clone().detach()
print(f"\nStep 0 (초기):")
print(f"  처음 10개 query 예측: {initial_points[:, 0].mean():.4f}, {initial_points[:, 1].mean():.4f}")
print(f"  Std: {initial_points[:, 0].std():.4f}, {initial_points[:, 1].std():.4f}")

for step in range(10):
    samples, targets = next(iter(train_loader))
    samples = samples.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    optimizer.zero_grad()
    outputs = model(samples)
    loss_dict = criterion(outputs, targets)
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    losses.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    
    if (step + 1) % 5 == 0:
        with torch.no_grad():
            # 같은 샘플로 테스트
            samples_test, _ = next(iter(val_loader))
            samples_test = samples_test.to(device)
            test_outputs = model(samples_test)
            test_points = test_outputs['pred_points'][0, :10]
            
            print(f"\nStep {step+1}:")
            print(f"  Loss: {losses.item():.4f} (CE: {loss_dict['loss_ce'].item():.4f}, Pt: {loss_dict['loss_point'].item():.4f})")
            print(f"  처음 10개 query 예측: {test_points[:, 0].mean():.4f}, {test_points[:, 1].mean():.4f}")
            print(f"  Std: {test_points[:, 0].std():.4f}, {test_points[:, 1].std():.4f}")

print(f"\n" + "="*80)
print("📊 진단 요약")
print("="*80)
print(f"\n1. 타겟 라벨: 위에서 확인 → 다양한 위치라면 ✅")
print(f"2. 초기 예측: std < 0.01이면 모든 query가 (0.5, 0.5) 근처")
print(f"3. 학습 후: std가 증가했다면 query 분화 시작 ✅")
print(f"\n✅ 해결책:")
print(f"   - Std < 0.01은 초기화 때문 (정상)")
print(f"   - 학습을 계속하면 분화됨")
print(f"   - 만약 100 steps 후에도 std < 0.01이면:")
print(f"     → Learning rate 너무 작음")
print(f"     → Backbone frozen 확인")
print(f"     → Loss weight 확인")
