import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import json

from config.config import load_config
from models import get_model
from utils.enhanced_tta import EnhancedTTA, create_tta_ensemble_predictions
from utils.advanced_ensemble import (
    WeightedEnsemble, 
    VotingEnsemble, 
    AdaptiveEnsemble,
    create_ensemble_predictions
)


class TestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_id = self.data.iloc[idx]['ID']
        img_path = os.path.join(self.img_dir, image_id)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = np.array(image)
            image = self.transform(image=image)['image']
            
        return image, image_id


def load_models(model_configs, device):
    """
    여러 모델을 로드하는 함수
    """
    models = []
    
    for config in model_configs:
        model_class = get_model(config['model_name'])
        model = model_class(num_classes=config['num_classes']).to(device)
        
        # 체크포인트 로드
        checkpoint = torch.load(config['checkpoint_path'], map_location=device)
        try:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except RuntimeError as e:
            print(f"Error loading {config['checkpoint_path']}: {e}")
            print("Attempting to load with strict=False")
            try:
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            except RuntimeError as e2:
                print(f"Failed to load checkpoint {config['checkpoint_path']}: {e2}")
                continue
        
        model.eval()
        print(f"모델 {config['model_name']} 훈련 모드: {model.training}")
        models.append(model)
        print(f"모델 로드 완료: {config['model_name']} from {config['checkpoint_path']}")
    
    return models


def create_test_transform(image_size=(640, 640)):
    """
    테스트용 변환 생성
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def enhanced_inference(
    models, 
    test_loader, 
    device, 
    ensemble_type='weighted',
    weights=None,
    use_tta=True,
    num_tta=5,
    use_adaptive_tta=False,
    confidence_threshold=0.9
):
    """
    향상된 추론 함수
    
    Args:
        models: 모델 리스트
        test_loader: 테스트 데이터 로더
        device: 디바이스
        ensemble_type: 앙상블 타입
        weights: 가중치
        use_tta: TTA 사용 여부
        num_tta: TTA 개수
        use_adaptive_tta: 적응형 TTA 사용 여부
        confidence_threshold: 신뢰도 임계값
    
    Returns:
        예측 결과와 이미지 이름 리스트
    """
    all_predictions = []
    all_image_ids = []
    
    # TTA 객체 생성
    if use_tta:
        tta = EnhancedTTA()
    
    # 모든 모델을 eval 모드로 설정
    for model in models:
        model.eval()
    
    # 앙상블 객체 생성
    if ensemble_type == 'weighted':
        ensemble = WeightedEnsemble(models, weights)
    elif ensemble_type == 'voting':
        ensemble = VotingEnsemble(models, voting='soft')
    elif ensemble_type == 'adaptive':
        ensemble = AdaptiveEnsemble(models, confidence_threshold)
    else:
        raise ValueError(f"지원하지 않는 앙상블 타입: {ensemble_type}")
    
    print(f"추론 시작 - 앙상블: {ensemble_type}, TTA: {use_tta}")
    
    for batch_idx, (images, image_ids) in enumerate(tqdm(test_loader, desc="추론 중")):
        images = images.to(device)
        
        if use_tta:
            if use_adaptive_tta:
                # 적응형 TTA
                batch_predictions = []
                for i in range(len(images)):
                    img_predictions = []
                    for model in models:
                        model.eval()  # 확실히 eval 모드로 설정
                        # 각 이미지에 대해 적응형 TTA 적용
                        img_np = images[i].cpu().numpy().transpose(1, 2, 0)
                        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                        img_np = np.clip(img_np, 0, 1)
                        img_np = (img_np * 255).astype(np.uint8)
                        
                        from utils.enhanced_tta import AdaptiveTTA
                        adaptive_tta = AdaptiveTTA(confidence_threshold=confidence_threshold)
                        pred, num_used = adaptive_tta.predict_with_adaptive_tta(model, img_np, device)
                        img_predictions.append(pred)
                    
                    # 모델 평균
                    avg_pred = torch.stack(img_predictions).mean(dim=0)
                    batch_predictions.append(avg_pred)
                
                batch_predictions = torch.stack(batch_predictions)
            else:
                # 일반 TTA
                batch_predictions = []
                for model in models:
                    model.eval()  # 확실히 eval 모드로 설정
                    model_pred = tta.predict_batch_with_tta(model, images, device, num_tta)
                    batch_predictions.append(model_pred)
                
                # 모델 평균
                batch_predictions = torch.stack(batch_predictions).mean(dim=0)
        else:
            # TTA 없이 일반 앙상블
            batch_predictions = ensemble.predict(images, device)
        
        all_predictions.append(batch_predictions.cpu())
        all_image_ids.extend(image_ids)
    
    # 모든 예측 결과 합치기
    all_predictions = torch.cat(all_predictions, dim=0)
    
    return all_predictions, all_image_ids


def save_predictions(predictions, image_ids, class_names, save_path):
    """
    예측 결과를 CSV 파일로 저장 (pred.csv와 동일한 형식)
    """
    # 가장 높은 확률의 클래스 선택
    predicted_classes = torch.argmax(predictions, dim=1).numpy()
    
    # 결과 DataFrame 생성 (pred.csv와 동일한 형식: ID, target)
    results = pd.DataFrame({
        'ID': image_ids,
        'target': predicted_classes
    })
    
    # 저장
    results.to_csv(save_path, index=False)
    print(f"예측 결과 저장 완료: {save_path}")
    
    return results


def analyze_predictions(predictions, image_ids, class_names):
    """
    예측 결과 분석
    """
    probs = torch.softmax(predictions, dim=1).numpy()
    predicted_classes = torch.argmax(predictions, dim=1).numpy()
    max_probs = np.max(probs, axis=1)
    
    print("\n=== 예측 결과 분석 ===")
    print(f"총 예측 수: {len(predictions)}")
    print(f"평균 신뢰도: {np.mean(max_probs):.4f}")
    print(f"최소 신뢰도: {np.min(max_probs):.4f}")
    print(f"최대 신뢰도: {np.max(max_probs):.4f}")
    
    # 클래스별 예측 분포
    print("\n클래스별 예측 분포:")
    unique, counts = np.unique(predicted_classes, return_counts=True)
    for class_idx, count in zip(unique, counts):
        print(f"{class_names[class_idx]}: {count} ({count/len(predictions)*100:.1f}%)")
    
    # 낮은 신뢰도 예측 분석
    low_confidence_threshold = 0.7
    low_confidence_mask = max_probs < low_confidence_threshold
    low_confidence_count = np.sum(low_confidence_mask)
    
    if low_confidence_count > 0:
        print(f"\n낮은 신뢰도 예측 (< {low_confidence_threshold}): {low_confidence_count}개")
        print("낮은 신뢰도 이미지 목록:")
        for i, is_low in enumerate(low_confidence_mask):
            if is_low:
                print(f"  {image_ids[i]}: {class_names[predicted_classes[i]]} (신뢰도: {max_probs[i]:.3f})")


def main():
    # 설정 로드
    cfg = load_config("config/main_config.yaml")
    
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")
    
    # 클래스 이름 로드
    meta_df = pd.read_csv("./data/meta_kr.csv")
    class_names = meta_df["class_name"].tolist()
    
    # 모델 설정 (여러 모델 사용 가능)
    model_configs = [
        {
            'model_name': 'ResNeSt101eModelArcFaceModel', # main_kfold.py 결과
            'checkpoint_path': './output/final_model_kfold.pth',
            'num_classes': 17
        },
        {
            'model_name': 'ConvNeXtV2Model', # test_convnext_v2.py 결과
            'checkpoint_path': './output/convnext_v2_test.pth',
            'num_classes': 17
        },
        {
            'model_name': 'EfficientNetV2MArcFaceModel', # main.py에서 새로 훈련한 모델
            'checkpoint_path': './output/efficientnet_v2_checkpoint.pth',
            'num_classes': 17
        },
        # {
        #     'model_name': 'ResNeSt101eModelArcFaceModel', # main.py 결과
        #     'checkpoint_path': './output/checkpoint.pth',
        #     'num_classes': 17
        # }
        # 추가 모델이 있다면 여기에 추가
        # {
        #     'model_name': 'ConvNeXtV2Model',
        #     'checkpoint_path': './output/convnext_v2.pth',
        #     'num_classes': 17
        # }
    ]
    
    # 모델 로드
    models = load_models(model_configs, device)
    
    # 테스트 데이터 로드
    test_transform = create_test_transform()
    test_dataset = TestDataset(
        csv_file='./data/sample_submission.csv',
        img_dir='./data/test/',
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['BATCH_SIZE'] * 2,  # 추론 시에는 더 큰 배치 크기 사용
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 향상된 추론 수행
    predictions, image_ids = enhanced_inference(
        models=models,
        test_loader=test_loader,
        device=device,
        ensemble_type='weighted',  # 'weighted', 'voting', 'adaptive'
        weights=[0.4, 0.3, 0.3],  # K-Fold 모델에 높은 가중치 부여
        use_tta=True,
        num_tta=7, # 화질 개선 TTA를 포함하여 개수 증가
        use_adaptive_tta=False,
        confidence_threshold=0.9
    )
    
    # 예측 결과 분석
    analyze_predictions(predictions, image_ids, class_names)
    
    # 결과 저장
    save_path = './enhanced_predictions.csv'
    results = save_predictions(predictions, image_ids, class_names, save_path)
    
    print(f"\n추론 완료! 결과 파일: {save_path}")


if __name__ == "__main__":
    main()