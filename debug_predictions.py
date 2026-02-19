"""
DETR 예측 디버깅 스크립트
모든 query가 같은 위치를 예측하는 문제 진단
"""
import torch
import numpy as np

def debug_model_predictions(model, dataset, device, num_samples=3):
    """
    모델의 예측을 상세히 분석
    """
    model.eval()
    
    print("="*80)
    print("🔍 DETR Prediction Debugging")
    print("="*80)
    
    with torch.no_grad():
        for idx in range(num_samples):
            image_tensor, target = dataset[idx]
            
            # 배치 차원 추가
            from Deformable_DETR_train import nested_tensor_from_tensor_list
            image_batch = nested_tensor_from_tensor_list([image_tensor]).to(device)
            
            # 예측
            outputs = model(image_batch)
            
            logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
            points = outputs['pred_points'][0]  # [num_queries, 2]
            
            probs = logits.sigmoid()
            max_probs, pred_classes = probs.max(dim=-1)
            
            # 통계 계산
            print(f"\n📊 Sample {idx + 1}:")
            print(f"  Ground Truth: {len(target['points'])} cells")
            print(f"\n  All Predictions (num_queries={len(points)}):")
            print(f"    Point predictions - Mean: {points.mean(dim=0).cpu().numpy()}, Std: {points.std(dim=0).cpu().numpy()}")
            print(f"    Max probability - Mean: {max_probs.mean():.4f}, Max: {max_probs.max():.4f}, Min: {max_probs.min():.4f}")
            
            # Confidence별 필터링
            for threshold in [0.1, 0.3, 0.5, 0.7]:
                keep = max_probs > threshold
                num_keep = keep.sum().item()
                
                if num_keep > 0:
                    kept_points = points[keep].cpu().numpy()
                    kept_probs = max_probs[keep].cpu().numpy()
                    
                    # 점들이 얼마나 다양한지 확인
                    point_std = kept_points.std(axis=0)
                    point_range = kept_points.max(axis=0) - kept_points.min(axis=0)
                    
                    print(f"\n  Threshold {threshold}:")
                    print(f"    Kept: {num_keep} predictions")
                    print(f"    Point diversity - Std: [{point_std[0]:.4f}, {point_std[1]:.4f}], Range: [{point_range[0]:.4f}, {point_range[1]:.4f}]")
                    
                    if num_keep <= 10:
                        print(f"    Actual points (x, y, prob):")
                        for i, (pt, prob) in enumerate(zip(kept_points, kept_probs)):
                            print(f"      {i+1}. ({pt[0]:.3f}, {pt[1]:.3f}) - prob: {prob:.4f}")
                else:
                    print(f"\n  Threshold {threshold}: 0 predictions")
            
            print("\n" + "-"*80)
    
    print("\n" + "="*80)
    print("🔍 Debugging Complete")
    print("="*80)

if __name__ == "__main__":
    print("Run this in notebook to debug predictions")
