import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as T

from config.config import load_config
from models import get_model
from datasets import get_dataset
from datasets.transforms import build_unified_transforms
from utils.predict_tta import get_tta_predictions


# --- 기본 설정 ---
cfg = load_config("config/inference_config.yaml")
ModelClass_A = get_model(cfg['MODEL_A']['name'])
ModelClass_B = get_model(cfg['MODEL_B']['name'])
ModelClass_C = get_model(cfg['MODEL_C']['name'])
ModelClass_D = get_model(cfg['MODEL_D']['name'])
ModelClassList = [
    ModelClass_A,
    ModelClass_B,
    ModelClass_C,
]
DatasetClass = get_dataset(cfg['DATASET'])
num_classes = 17
num_workers = os.cpu_count() // 2
output_root = './output'
data_path = './data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_transform = build_unified_transforms(cfg["transforms"]["test"])
tst_dataset = DatasetClass(
    # f"{data_path}/sample_submission.csv",
    f"{data_path}/processed_ensemble_pred.csv",
    # f"{data_path}/test/",
    f"{data_path}/test_modify/",
    transform=test_transform
)
tst_loader = DataLoader(tst_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=num_workers)

# --- 앙상블 설정 ---
# 1. 사용할 모델들의 체크포인트 경로 리스트
model_paths = [
    f"{output_root}/EfficientNetV2MArcFaceModel_2025-07-05_20-40_checkpoint.pth",
    f"{output_root}/ResNeStModelArcFaceModel_2025-07-06_03-13_checkpoint.pth",
    f"{output_root}/ConvNeXtArcFaceModel_2025-07-06_05-22_checkpoint.pth",
]

# 가중치 리스트 정의
# 모델 경로(model_paths) 순서와 동일하게 가중치를 설정합니다.
# 가중치의 합은 1.0이 되어야 합니다.
num_models = len(model_paths)
use_weight = False
if use_weight:
    weights = [0.2, 0.4, 0.4]
    # weights = [0.2, 0.3, 0.3, 0.2]
else:
    weights = [1 / num_models] * num_models

# 2. TTA 변환 정의
tta_transforms = [
    T.RandomHorizontalFlip(p=1.0),
    T.RandomAffine(degrees=10, scale=(0.9, 1.1)),
]

# --- 모델 로드 ---
ensemble_models = []
print("앙상블 모델 로딩 시작...")
for path, ModelClass in zip(model_paths, ModelClassList):
    # 주의: 모델 아키텍처가 다를 경우, get_model 등을 통해 각 모델에 맞는 클래스를 불러와야 합니다.
    model = ModelClass(num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval() # 평가 모드로 즉시 설정
    ensemble_models.append(model)
print(f"{len(ensemble_models)}개의 모델 로딩 완료.")

# --- 앙상블 + TTA 추론 루프 ---
final_predictions = []
for images, _, _ in tqdm(tst_loader, desc="Weighted Ensemble Inferencing"):
    
    batch_model_predictions = []

    # 각 모델에 대해 TTA 추론 수행
    for model in ensemble_models:
        model_avg_probs = get_tta_predictions(model, images, tta_transforms, device)
        batch_model_predictions.append(model_avg_probs)
    
    # 1. 예측 결과들을 하나의 텐서로 합칩니다. (형태: [모델개수, 배치크기, 클래스개수])
    stacked_preds = torch.stack(batch_model_predictions, dim=0)
    
    # 2. 가중치를 텐서로 변환하고, 브로드캐스팅을 위해 차원을 변경합니다.
    # [3] -> [3, 1, 1] 형태로 변경하여 stacked_preds와 곱셈이 가능하도록 합니다.
    weight_tensor = torch.tensor(weights, device=device).view(-1, 1, 1)
    
    # 3. 가중치를 곱하여 가중 평균을 계산합니다.
    # stacked_preds와 weight_tensor를 곱한 뒤, 모델 차원(dim=0)에 대해 합산합니다.
    weighted_ensemble_probs = (stacked_preds * weight_tensor).sum(dim=0)
    
    # 최종 클래스 예측
    final_class_preds = weighted_ensemble_probs.argmax(dim=1)
    
    final_predictions.extend(final_class_preds.detach().cpu().numpy())
    
# --- 결과 저장 (기존 코드와 동일) ---
print("추론 완료. CSV 파일 생성 중...")
pred_df = pd.DataFrame({'ID': tst_dataset.df['ID'], 'target': final_predictions})
pred_df.to_csv("pred_ensemble.csv", index=False)
print("pred_ensemble.csv 파일 저장 완료!")