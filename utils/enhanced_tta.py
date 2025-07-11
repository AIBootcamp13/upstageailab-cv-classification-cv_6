import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class EnhancedTTA:
    """
    Enhanced Test Time Augmentation for Document Classification
    """
    
    def __init__(self, image_size=(640, 640), num_classes=17):
        self.image_size = image_size
        self.num_classes = num_classes
        
        # TTA 변환 정의
        self.tta_transforms = self._create_tta_transforms()
        
    def _create_tta_transforms(self):
        """
        문서 분류에 특화된 TTA 변환들을 생성
        """
        transforms_list = []
        
        # 1. 원본 (변환 없음)
        # 원본 이미지는 predict_batch_with_tta 함수에서 별도로 처리하므로 여기서는 TTA 변환만 정의합니다.
        
        # 1. 수평 뒤집기 (문서 좌우 반전은 내용에 영향을 주지 않는 경우가 많음)
        transforms_list.append(
            A.Compose([
                A.HorizontalFlip(p=1.0),
            ])
        )
        
        # 2. 미세한 회전 (문서가 약간 기울어져 스캔된 경우를 모방)
        transforms_list.append(
            A.Compose([
                A.Rotate(limit=15, p=1.0),
            ])
        )
        
        # 3. 크기 조절 (문서가 다른 크기로 스캔된 경우를 모방)
        transforms_list.append(
            A.Compose([
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=0, p=1.0),
            ])
        )
        
        # 4. 원근 변환 (문서를 비스듬히 촬영한 경우를 모방)
        transforms_list.append(
            A.Compose([
                A.Perspective(scale=(0.05, 0.1), p=1.0),
            ])
        )
        
        # 5. 밝기/대비 조절 (다양한 조명 환경을 모방)
        transforms_list.append(
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            ])
        )
        
        # 6. 선명도 강화 (흐릿한 이미지를 보완)
        transforms_list.append(
            A.Compose([
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
            ])
        )
        
        # 7. 해상도 저하/복원 (압축/저화질 환경 모방)
        transforms_list.append(
            A.Compose([
                A.Downscale(scale_min=0.5, scale_max=0.8, interpolation=cv2.INTER_AREA, p=1.0),
            ])
        )

        # 모든 변환에 공통적으로 적용될 후처리 단계
        post_transforms = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # 각 TTA 변환과 후처리 단계를 결합
        transforms_list = [A.Compose([t, post_transforms]) for t in transforms_list]

        return transforms_list
    
    def predict_with_tta(self, model, image, device, use_all_transforms=True):
        """
        TTA를 적용하여 예측 수행
        
        Args:
            model: 학습된 모델
            image: PIL Image 또는 numpy array
            device: 디바이스
            use_all_transforms: 모든 변환 사용 여부
        
        Returns:
            평균 예측 확률
        """
        model.eval()
        
        # 이미지가 PIL Image인 경우 numpy array로 변환
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))
        
        all_predictions = []
        
        # 사용할 변환 선택
        transforms_to_use = self.tta_transforms if use_all_transforms else self.tta_transforms[:5]
        
        with torch.no_grad():
            for transform in transforms_to_use:
                # 변환 적용
                transformed = transform(image=image)['image']
                
                # 배치 차원 추가
                if transformed.dim() == 3:
                    transformed = transformed.unsqueeze(0)
                
                # 디바이스로 이동
                transformed = transformed.to(device)
                
                # 예측
                output = model(transformed)
                
                # 소프트맥스 적용
                probs = F.softmax(output, dim=1)
                all_predictions.append(probs)
        
        # 모든 예측의 평균 계산
        avg_prediction = torch.stack(all_predictions).mean(dim=0)
        
        return avg_prediction
    
    def predict_batch_with_tta(self, model, image_batch, device, num_tta=5):
        """
        배치에 대해 TTA 예측 수행
        
        Args:
            model: 학습된 모델
            image_batch: 이미지 배치 텐서
            device: 디바이스
            num_tta: 사용할 TTA 개수
        
        Returns:
            평균 예측 확률
        """
        model.eval()
        
        batch_size = image_batch.size(0)
        all_predictions = []
        
        with torch.no_grad():
            # 원본 이미지 예측
            original_output = model(image_batch)
            original_probs = F.softmax(original_output, dim=1)
            all_predictions.append(original_probs)
            
            # TTA 적용
            # BUG FIX: `len(self.tta_transforms) - 1`로 인해 마지막 변환이 누락되는 버그 수정
            # num_tta와 실제 변환 개수 중 작은 값을 기준으로 반복합니다.
            num_transforms_to_apply = min(num_tta, len(self.tta_transforms))
            
            for i in range(num_transforms_to_apply):
                transform = self.tta_transforms[i]
                # 배치의 각 이미지에 변환 적용
                # 참고: 이 루프는 현재 구조에서 비효율적이지만, 정확한 TTA 적용을 위해 필요합니다.
                tta_batch = []
                for j in range(batch_size):
                    # GPU Tensor -> CPU Numpy 변환
                    img_np = image_batch[j].cpu().numpy().transpose(1, 2, 0)
                    
                    # 정규화 해제
                    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    img_np = np.clip(img_np, 0, 1)
                    img_np = (img_np * 255).astype(np.uint8)
                    
                    # 변환 적용
                    transformed = transform(image=img_np)['image']
                    tta_batch.append(transformed)
                
                # 배치로 합치기
                tta_batch = torch.stack(tta_batch).to(device)
                
                # 예측
                tta_output = model(tta_batch)
                tta_probs = F.softmax(tta_output, dim=1)
                all_predictions.append(tta_probs)
        
        # 모든 예측의 평균 계산
        avg_prediction = torch.stack(all_predictions).mean(dim=0)
        
        return avg_prediction


class AdaptiveTTA:
    """
    적응형 TTA - 모델 신뢰도에 따라 TTA 강도 조절
    """
    
    def __init__(self, image_size=(640, 640), confidence_threshold=0.9):
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.enhanced_tta = EnhancedTTA(image_size)
        
    def predict_with_adaptive_tta(self, model, image, device):
        """
        적응형 TTA 예측
        
        Args:
            model: 학습된 모델
            image: 입력 이미지
            device: 디바이스
        
        Returns:
            예측 확률과 사용된 TTA 개수
        """
        model.eval()
        
        # 첫 번째 변환 (원본)으로 초기 예측
        transform = self.enhanced_tta.tta_transforms[0]
        
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))
        
        with torch.no_grad():
            transformed = transform(image=image)['image'].unsqueeze(0).to(device)
            initial_output = model(transformed)
            initial_probs = F.softmax(initial_output, dim=1)
            
            # 최대 신뢰도 확인
            max_confidence = initial_probs.max().item()
            
            if max_confidence >= self.confidence_threshold:
                # 신뢰도가 높으면 간단한 TTA만 적용
                return self.enhanced_tta.predict_with_tta(
                    model, image, device, use_all_transforms=False
                ), 5
            else:
                # 신뢰도가 낮으면 전체 TTA 적용
                return self.enhanced_tta.predict_with_tta(
                    model, image, device, use_all_transforms=True
                ), 10


def create_tta_ensemble_predictions(models, image_batch, device, num_tta=5):
    """
    다중 모델과 TTA를 결합한 앙상블 예측
    
    Args:
        models: 모델 리스트
        image_batch: 이미지 배치
        device: 디바이스
        num_tta: TTA 개수
    
    Returns:
        앙상블 예측 확률
    """
    enhanced_tta = EnhancedTTA()
    all_model_predictions = []
    
    for model in models:
        # 각 모델에 대해 TTA 적용
        model_predictions = enhanced_tta.predict_batch_with_tta(
            model, image_batch, device, num_tta
        )
        all_model_predictions.append(model_predictions)
    
    # 모든 모델의 예측 평균
    ensemble_prediction = torch.stack(all_model_predictions).mean(dim=0)
    
    return ensemble_prediction