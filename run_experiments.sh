#!/bin/bash

echo "=== 문서 분류 F1 macro 점수 개선 실험 ==="

# 1. 현재 기본 모델 + Label Smoothing 테스트 (5 epoch)
echo "1단계: 기본 모델 + Label Smoothing 테스트"
python main.py

# 2. ConvNeXt-V2 모델 테스트 (10 epoch)
echo "2단계: ConvNeXt-V2 모델 테스트"
python test_convnext_v2.py

# 3. K-Fold 교차검증 (가장 좋은 모델로)
echo "3단계: K-Fold 교차검증"
python main_kfold.py

# 4. 향상된 추론 테스트
echo "4단계: 향상된 추론 (TTA + 앙상블)"
python inference_enhanced.py

echo "=== 모든 실험 완료 ==="