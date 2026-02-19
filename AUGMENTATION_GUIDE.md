# HnE Cell Detection - Data Augmentation Guide

## 🎨 적용된 증강 기법 (Applied Augmentation Techniques)

### 1. 기하학적 증강 (Geometric Augmentation)
병리 이미지의 방향성은 중요하지 않으므로, 다양한 회전과 플립을 적용합니다.

- **Horizontal Flip** (50% probability)
  - 좌우 반전
  - Point coordinates 자동 조정

- **Vertical Flip** (50% probability)
  - 상하 반전
  - Point coordinates 자동 조정

- **90° Rotation** (30% probability)
  - 90°, 180°, 270° 무작위 회전
  - Point coordinates 회전 행렬로 조정

---

### 2. 색상 증강 (Color Augmentation)
스캐너 및 병원별 염색 차이를 시뮬레이션합니다.

#### A. 밝기 조정 (Brightness Adjustment)
- **Range**: 0.8 ~ 1.2 (±20%)
- **목적**: 스캐너 노출 설정 차이
- **Probability**: 50%

#### B. 대비 조정 (Contrast Adjustment)
- **Range**: 0.8 ~ 1.2 (±20%)
- **목적**: 스캐너 대비 설정 차이
- **Probability**: 50%

#### C. 색조 이동 (Hue Shift)
- **Range**: -10° ~ +10°
- **목적**: 염색 색상 변화 (H&E 염색 배치별 차이)
- **Probability**: 50%
- **Method**: HSV 색공간에서 조정

#### D. 채도 조정 (Saturation Adjustment)
- **Range**: 0.7 ~ 1.3 (±30%)
- **목적**: 염색 강도 변화
- **Probability**: 50%
- **Method**: HSV 색공간에서 조정

#### E. 감마 보정 (Gamma Correction)
- **Range**: 0.8 ~ 1.2
- **목적**: 스캐너 감마 곡선 차이
- **Probability**: 30%

#### F. RGB 채널 이동 (RGB Channel Shift)
- **Range**: -10 ~ +10 (각 채널)
- **목적**: 색보정 차이
- **Probability**: 30%

---

### 3. 노이즈 및 블러 (Noise & Blur)
스캐너의 물리적 특성 차이를 시뮬레이션합니다.

#### A. Gaussian Noise
- **Range**: 0 ~ 5 (standard deviation)
- **목적**: 스캐너 센서 노이즈
- **Probability**: 30%

#### B. Gaussian Blur
- **Kernel Size**: 3x3 또는 5x5
- **목적**: 초점 변화
- **Probability**: 20%

---

## 🔬 병리 이미지 특화 증강의 중요성

### 왜 HnE 이미지는 특별한 증강이 필요한가?

1. **스캐너별 차이**
   - Aperio, Hamamatsu, Leica 등 스캐너마다 다른 색상 프로파일
   - 조명 시스템, 카메라 센서, 렌즈 특성 차이

2. **병원/실험실별 차이**
   - 염색 프로토콜 (시약, 시간, 농도)
   - 슬라이드 준비 과정
   - 보관 조건

3. **배치별 차이**
   - 같은 병원에서도 염색 배치마다 색상 변화
   - 시약 로트 변경
   - 시간에 따른 슬라이드 퇴색

---

## 📊 증강 효과 검증

### Training 시 기대 효과:
1. ✅ **Generalization**: 다양한 스캐너/병원 데이터에서 안정적 성능
2. ✅ **Robustness**: 색상 변화에 덜 민감한 모델
3. ✅ **Data Efficiency**: 제한된 데이터로 더 나은 성능

### 검증 방법:
- 다른 병원/스캐너 데이터로 테스트
- Cross-validation 성능 향상 확인
- 색상 변화에 대한 민감도 분석

---

## 🚀 고급 증강 기법 (Advanced Techniques)

`utils/stain_augmentation.py`에 다음 고급 기법들이 구현되어 있습니다:

### 1. Macenko Stain Normalization
- H&E 염색 분리 및 재조합
- 참조 이미지로 정규화 가능

### 2. HED Color Deconvolution
- Hematoxylin, Eosin, DAB 채널 분리
- 각 염색 성분을 독립적으로 조정

### 3. LAB Color Space Augmentation
- Perceptual uniform color space
- 더 자연스러운 색상 변화

---

## 💡 사용 권장사항

### Training:
```python
train_dataset = P2PNetDataset(
    train_images, 
    train_labels, 
    augment=True,  # 증강 활성화
    max_points=500
)
```

### Validation/Test:
```python
val_dataset = P2PNetDataset(
    val_images, 
    val_labels, 
    augment=False,  # 증강 비활성화
    max_points=500
)
```

---

## 📈 증강 전후 비교

증강을 시각화하려면:
```python
visualize_augmentation_effects(train_dataset, index=5, num_augmentations=6)
```

같은 이미지에 대해 6가지 다른 증강 버전을 생성하여,
스캐너/병원별 변화를 시뮬레이션한 결과를 확인할 수 있습니다.

---

## ⚠️ 주의사항

1. **Point Annotations 보존**: 
   - 기하학적 변환 시 point coordinates가 정확히 따라가는지 확인
   
2. **Over-augmentation 방지**: 
   - 너무 극단적인 증강은 오히려 성능 저하
   - 현재 설정은 실제 변화 범위 내에서 조정됨

3. **Validation에는 증강 사용 안 함**:
   - 공정한 성능 평가를 위해 원본 이미지 사용

---

## 📚 References

1. Macenko et al. (2009) - "A method for normalizing histology slides for quantitative analysis"
2. Tellez et al. (2018) - "Whole-Slide Mitosis Detection in H&E Breast Histology Using PHH3 as a Reference to Train Distilled Stain-Invariant Convolutional Networks"
3. Ciompi et al. (2017) - "The importance of stain normalization in colorectal tissue classification with convolutional networks"
