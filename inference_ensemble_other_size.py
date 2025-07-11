import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

from config.config import load_config
from models import get_model
from datasets import get_dataset
from datasets.transforms import build_unified_transforms
from utils.predict_tta import get_tta_predictions

# --- 기본 설정 ---
cfg = load_config("config/inference_config.yaml")
DatasetClass = get_dataset(cfg['DATASET'])
num_classes = 17
num_workers = os.cpu_count() // 2
data_path = './data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 앙상블 설정 (더 유연한 구조로 변경) ---
# 각 모델의 정보를 딕셔너리로 묶어 리스트로 관리합니다.
# 이렇게 하면 모델별로 다른 이미지 크기, 배치 사이즈, 가중치를 쉽게 설정할 수 있습니다.


model_configs = [
    {
        "name": "EfficientNetV2",
        "model_class": get_model(cfg['MODEL_A']['name']),
        "path": f"./output/EfficientNetV2MArcFaceModel_2025-07-05_20-40_checkpoint.pth",
        "transform_params": cfg["transforms"]["model_a_test"], # 예: { "size": 640 }
        "batch_size": 16,
    },
    {
        "name": "ResNeSt",
        "model_class": get_model(cfg['MODEL_B']['name']),
        "path": f"./output/ResNeStModelArcFaceModel_2025-07-06_03-13_checkpoint.pth",
        "transform_params": cfg["transforms"]["model_b_test"], # 예: { "size": 640 }
        "batch_size": 16,
    },
    {
        "name": "ConvNeXt",
        "model_class": get_model(cfg['MODEL_C']['name']),
        "path": f"./output/ConvNeXtArcFaceModel_2025-07-09_06-12_checkpoint.pth",
        "transform_params": cfg["transforms"]["model_c_test"], # 예: { "size": 448 }
        "batch_size": 16,
    }
]

# TTA 변환 정의
tta_transforms = [
    T.RandomHorizontalFlip(p=1.0),
    T.RandomAffine(degrees=10, scale=(0.9, 1.1)),
]

# --- 모델별 추론 및 결과 취합 루프 ---
all_model_probs = []
test_df = pd.read_csv(f"{data_path}/processed_ensemble_pred.csv")

for config in model_configs:
    model_name = config['name']
    print(f"--- {model_name} 모델 추론 시작 (이미지 크기: {config['transform_params']['Resize']}, 배치 사이즈: {config['batch_size']}) ---")
    
    # 1. 모델별 전용 Transform, Dataset, DataLoader 생성
    model_transform = build_unified_transforms(config["transform_params"]["test"])
    model_dataset = DatasetClass(
        df=test_df,
        img_dir=f"{data_path}/test_modify/",
        transform=model_transform
    )
    model_loader = DataLoader(
        model_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers
    )

    # 2. 모델 로드
    model = config["model_class"](num_classes).to(device)
    model.load_state_dict(torch.load(config["path"], map_location=device))
    model.eval()

    # 3. 현재 모델로 전체 데이터에 대한 예측 확률 계산
    model_single_predictions = []
    with torch.no_grad():
        for images, _, _ in tqdm(model_loader, desc=f"Inferencing with {model_name}"):
            # TTA를 적용한 평균 확률을 얻습니다.
            # get_tta_predictions 함수는 내부적으로 softmax를 적용하여 확률을 반환해야 합니다.
            model_avg_probs = get_tta_predictions(model, images, tta_transforms, device)
            model_single_predictions.append(model_avg_probs)
    
    # 현재 모델의 모든 예측 확률을 하나의 텐서로 합칩니다. (형태: [전체 데이터 수, 클래스 개수])
    model_total_probs = torch.cat(model_single_predictions, dim=0)
    all_model_probs.append(model_total_probs)


# --- 최종 앙상블 및 결과 저장 ---
print("\n--- 모든 모델 추론 완료. 최종 앙상블 수행 ---")

# 1. 모든 모델의 확률 예측값을 하나의 텐서로 합칩니다. (형태: [모델 개수, 전체 데이터 수, 클래스 개수])
stacked_probs = torch.stack(all_model_probs, dim=0).to(device)

# 2. 가중치 텐서를 생성하고 브로드캐스팅을 위해 차원을 변경합니다.
# [3] -> [3, 1, 1] 형태로 변경
weights = torch.tensor([config['weight'] for config in model_configs], device=device).view(-1, 1, 1)

# 3. 가중 평균을 계산합니다.
weighted_ensemble_probs = (stacked_probs * weights).sum(dim=0)

# 4. 최종 클래스 예측
final_predictions = weighted_ensemble_probs.argmax(dim=1).cpu().numpy()

# 5. 결과 저장
print("앙상블 완료. CSV 파일 생성 중...")
pred_df = pd.DataFrame({'ID': test_df['ID'], 'target': final_predictions})
pred_df.to_csv("pred_ensemble_multi_size.csv", index=False)
print("pred_ensemble_multi_size.csv 파일 저장 완료!")