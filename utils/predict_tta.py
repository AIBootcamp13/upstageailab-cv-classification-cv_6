from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F

def predict_with_tta(model: torch.nn.Module, image_path, base_transform, tta_transforms, device):
    """
    TTA를 적용하여 단일 이미지의 최종 예측 확률을 반환합니다.

    :param model: 학습 완료된 모델
    :param image_path: 예측할 이미지 경로
    :param base_transform: Resize, ToTensor, Normalize 등 기본 전처리 파이프라인
    :param tta_transforms: TTA에 사용할 증강 리스트
    :param device: cpu 또는 cuda
    """
    model.eval()
    
    # 1. 원본 이미지를 NumPy 배열로 로드
    image = np.array(Image.open(image_path).convert("RGB"))
    
    all_predictions = []

    with torch.no_grad():
        # 2. 원본 이미지에 대한 예측
        original_transformed = base_transform(image=image)['image'].unsqueeze(0).to(device)
        original_output = model(original_transformed)
        all_predictions.append(original_output)

        # 3. TTA가 적용된 이미지들에 대한 예측
        for tta_transform in tta_transforms:
            # TTA 증강 적용
            augmented_image = tta_transform(image=image)['image']
            
            # 기본 전처리 적용
            transformed_image = base_transform(image=augmented_image)['image'].unsqueeze(0).to(device)
            
            # 모델 예측
            output = model(transformed_image)
            all_predictions.append(output)
            
    # 4. 모든 예측 결과를 종합
    # 모든 로짓을 하나로 합침 (예: 3개의 TTA -> 4, 17 크기의 텐서)
    stacked_predictions = torch.stack(all_predictions, dim=0)
    
    # Softmax를 적용하여 확률로 변환 후 평균 계산
    avg_probabilities = F.softmax(stacked_predictions, dim=-1).mean(dim=0)
    
    return avg_probabilities




def get_tta_predictions(model: torch.nn.Module, image_batch, tta_transforms, device):
    """
    하나의 모델과 이미지 배치에 TTA를 적용하여 최종 예측 확률(softmax)을 반환합니다.
    
    :param model: 평가할 단일 모델
    :param image_batch: DataLoader로부터 받은 이미지 배치
    :param tta_transforms: TTA에 사용할 증강 리스트
    :param device: cpu 또는 cuda
    """
    # 모델을 평가 모드로 설정
    model.eval()
    
    # 원본 이미지와 TTA 적용 이미지들의 예측 확률을 저장할 리스트
    all_tta_probs = []

    with torch.no_grad():
        # 1. 원본 이미지에 대한 예측 확률
        image_batch = image_batch.to(device)
        original_probs = model(image_batch).softmax(dim=1)
        all_tta_probs.append(original_probs)

        # 2. TTA가 적용된 이미지들에 대한 예측 확률
        for tta_transform in tta_transforms:
            augmented_image = tta_transform(image_batch)
            tta_probs = model(augmented_image).softmax(dim=1)
            all_tta_probs.append(tta_probs)
            
    # 3. 모든 TTA 예측 확률의 평균 계산
    # (TTA개수+1, 배치크기, 클래스개수) -> (배치크기, 클래스개수)
    avg_probs = torch.stack(all_tta_probs).mean(dim=0)
    
    return avg_probs
