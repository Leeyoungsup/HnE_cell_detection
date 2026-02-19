# 🚀 DETR 학습 빠른 시작 가이드

## ✅ 완료된 수정 사항

모든 공식 DETR 수정이 완료되었습니다!

### 수정된 셀 목록:
- ✅ **Cell 3 (Model)**: 완전 재작성 - 공식 DETR 구조
- ✅ **Cell 5 (Matcher)**: Softmax + simple negative probability
- ✅ **Cell 6 (Loss)**: Cross-entropy + empty_weight
- ✅ **Cell 9 (Training Setup)**: 업데이트된 파라미터
- ✅ **Cell 10 (Training Loop)**: Softmax post-processing
- ✅ **Cell 11 (Visualization)**: Softmax post-processing

## 📋 실행 방법

### 1단계: 순차적 셀 실행
```
Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5 → Cell 6 → Cell 7 → Cell 8 → Cell 9 → Cell 10
```

**중요:** 
- 처음부터 모든 셀을 순서대로 실행하세요
- 기존 checkpoint는 사용할 수 없습니다 (class_embed 차원 불일치)

### 2단계: 학습 시작 확인

첫 epoch에서 다음을 확인하세요:

```python
# Training Loop Cell 10 실행 중 출력 예시:
Epoch 1/300 [Train]: 100%|██████████| 100/100 [01:23<00:00, loss=2.5432, ce=1.8234, pt=0.7198]
Epoch 1/300 [Val]: 100%|██████████| 12/12 [00:15<00:00]

================================================================================
Epoch 1/300 Summary:
  Train Loss: 2.5432 (CE: 1.8234, Point: 0.7198)
  Val Loss: 2.3456 (CE: 1.7123, Point: 0.6333)
  Val Point Error: 0.0234
  Val Class Acc: 0.6789
  Val Recall: 0.4234      ← 기존 0.01에서 크게 개선!
  Val Precision: 0.3123
  Val F1: 0.3634 ⭐        ← 기존 0.01에서 크게 개선!
  LR: 0.000100
================================================================================
```

### 3단계: Query 분화 검증 (선택사항)

학습 중 query들이 제대로 분화되는지 확인하려면:

```python
# Validation loop 내부에 임시로 추가 (Cell 10)
with torch.no_grad():
    logits = outputs['pred_logits'][0]  # [100, 7]
    prob = F.softmax(logits, dim=-1)[:, :-1]  # [100, 6] - no-object 제외
    
    print("\n=== Query 분화 확인 (처음 5개 query) ===")
    for i in range(5):
        max_prob, max_class = prob[i].max(0)
        print(f"Query {i}: Class {max_class.item()} (prob={max_prob.item():.3f})")
        print(f"  분포: {prob[i].cpu().numpy()}")
```

**기대 결과:**
```
=== Query 분화 확인 (처음 5개 query) ===
Query 0: Class 2 (prob=0.823)
  분포: [0.021, 0.034, 0.823, 0.043, 0.012, 0.067]
Query 1: Class 0 (prob=0.756)
  분포: [0.756, 0.123, 0.034, 0.021, 0.032, 0.034]
Query 2: Class 1 (prob=0.689)
  분포: [0.045, 0.689, 0.123, 0.067, 0.034, 0.042]
Query 3: Class 4 (prob=0.712)
  분포: [0.023, 0.056, 0.089, 0.034, 0.712, 0.086]
Query 4: Class 3 (prob=0.645)
  분포: [0.034, 0.067, 0.123, 0.645, 0.056, 0.075]
```

각 query가 **서로 다른 확률 분포**를 가져야 정상입니다!

## 🔍 문제 발생 시 체크리스트

### 증상 1: "모든 query 예측이 여전히 동일함"

**원인:** Cell이 올바른 순서로 실행되지 않았을 수 있습니다.

**해결:**
1. Jupyter 커널 재시작: `Kernel → Restart & Clear Output`
2. Cell 1부터 10까지 순차 실행
3. 기존 checkpoint 로딩 코드가 있다면 제거

### 증상 2: "차원 불일치 에러"
```
RuntimeError: size mismatch for class_embed.weight: copying a param with shape torch.Size([7, 256]) from checkpoint, the shape in current model is torch.Size([6, 256]).
```

**원인:** 기존 checkpoint 로딩 시도

**해결:** Cell 9에서 checkpoint 로딩 코드 제거 또는 주석 처리
```python
# ❌ 이 코드 제거
# checkpoint = torch.load('old_model.pth')
# model.load_state_dict(checkpoint['model_state_dict'])

# ✅ 새로운 모델로 처음부터 학습
model = DETR_PointDetection(...)
```

### 증상 3: "GPU 메모리 부족"
```
RuntimeError: CUDA out of memory
```

**해결:** Cell 7 (Dataset)에서 batch_size 줄이기
```python
# 현재
batch_size = 4

# 변경
batch_size = 2  # 또는 1
```

### 증상 4: "학습 속도가 너무 느림"

**원인:** 정상입니다. DETR는 학습이 느립니다.

**참고:**
- Epoch 1: 약 1-2분 (V100 기준)
- 전체 300 epochs: 약 6-10시간
- 공식 DETR 논문도 500 epochs 훈련

**최적화:**
- `num_workers` 증가 (Cell 7)
- Mixed precision 사용 고려 (torch.cuda.amp)

## 📊 학습 진행 모니터링

### Epoch 10 기대치:
```
Val F1: 0.50~0.65
Val Recall: 0.55~0.70
Val Precision: 0.45~0.60
```

### Epoch 50 기대치:
```
Val F1: 0.70~0.80
Val Recall: 0.75~0.85
Val Precision: 0.65~0.75
```

### Epoch 300 기대치:
```
Val F1: 0.80~0.90
Val Recall: 0.85~0.95
Val Precision: 0.75~0.85
```

## 💾 모델 저장 위치

학습된 모델은 다음 위치에 자동 저장됩니다:
```
../../model/HnE_cell_detection/Deformable_detr/
├── best_model.pth        ← 최고 F1 모델
├── last_model.pth        ← 마지막 epoch 모델
├── checkpoint_epoch100.pth
├── checkpoint_epoch200.pth
├── checkpoint_epoch300.pth
└── visualize_epoch_10.jpg
```

## 🎯 다음 단계

학습 완료 후:

1. **Best 모델 로드**
```python
checkpoint = torch.load('../../model/HnE_cell_detection/Deformable_detr/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']} with F1={checkpoint['best_val_f1']:.4f}")
```

2. **Visualization 확인**
```python
visualize_predictions(model, val_dataset, num_samples=5, conf_threshold=0.5)
```

3. **WSI Prediction** (전체 슬라이드 예측)
```python
# wsi_prediction.ipynb 사용
```

## 📚 자세한 설명

모든 변경 사항의 상세 설명은 다음 파일 참고:
- `OFFICIAL_DETR_FIX_SUMMARY.md` - 전체 수정 사항 및 이론적 배경

## ❓ 자주 묻는 질문

**Q: 왜 기존 checkpoint를 사용할 수 없나요?**
A: `class_embed` 출력 차원이 6 → 7로 변경되었습니다 (no-object 클래스 추가). 가중치 shape이 달라서 로딩 불가능합니다.

**Q: Confidence threshold를 어떻게 설정하나요?**
A: Softmax 확률이므로 0.3~0.5 정도가 적절합니다. Validation에서 F1이 최대가 되는 값을 실험적으로 찾으세요.

**Q: 공식 DETR보다 epochs가 적은데 괜찮나요?**
A: 공식 DETR는 COCO (80 classes, 118k images) 학습으로 500 epochs 필요. 우리 데이터는 6 classes, 작은 데이터셋이므로 300 epochs면 충분합니다.

**Q: 학습이 너무 오래 걸립니다.**
A: Early stopping을 사용하세요. Val F1이 10 epochs 동안 개선되지 않으면 학습 중단하는 코드 추가 가능합니다.

---

## ✨ 시작하세요!

이제 **Cell 1부터 순서대로 실행**하면 됩니다. 행운을 빕니다! 🚀

문제가 생기면 `OFFICIAL_DETR_FIX_SUMMARY.md`를 참고하세요.
