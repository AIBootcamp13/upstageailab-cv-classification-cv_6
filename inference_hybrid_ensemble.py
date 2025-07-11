import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn.functional as F
import torchvision.transforms as T

from utils.utils import set_seed
from config.config import load_config
from models import get_model
from datasets import get_dataset
from datasets.transforms import build_unified_transforms


# --- 1. 설정---

SEED = 777
set_seed(SEED)

cfg_main = load_config("config/main_config.yaml")
cfg_inf = load_config("config/inference_config.yaml")

# 기타 설정
INFERENCE_BATCH_SIZE = 16 # 추론 배치 사이즈 통일
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = './data'
output_root = './output'
num_classes = 17
num_workers = os.cpu_count() // 2
# test_csv_path = f"{data_path}/sample_submission.csv"
test_csv_path = f"{data_path}/processed_ensemble_pred.csv"
# test_img_dir = f"{data_path}/test/"
test_img_dir = f"{data_path}/test_modify/"




# 앙상블에 포함시킬 모델 '그룹'들을 리스트에 정의합니다.
ENSEMBLE_GROUPS = [
    {
        "group_name": "ConvNeXt_KFold_Ensemble",
        "type": "kfold",
        "model_class": get_model("ConvNeXtArcFaceModel"),
        "experiment_name": "ConvNeXtArcFaceModel_2025-07-09_15-17",
        "n_splits": 5,
        "transform_params": cfg_inf["transforms"]["test"],
        "group_weight": 0.2
    },
    {
        "group_name": "ConvNeXt_Single_Best",
        "type": "individual",
        "model_class": get_model("ConvNeXtArcFaceSingleModel"),
        "model_paths": [
            f"{output_root}/ConvNeXtArcFaceModel_2025-07-06_05-22_checkpoint.pth"
        ],
        "transform_params": cfg_inf["transforms"]["model_a_test"]["test"],
        "group_weight": 0.4
    },
    {
        "group_name": "ResNeSt50_Single_Best",
        "type": "individual",
        "model_class": get_model("ResNeSt50ArcFaceModel"),
        "model_paths": [
            f"{output_root}/ResNeStModelArcFaceModel_2025-07-06_03-13_checkpoint.pth"
        ],
        "transform_params": cfg_inf["transforms"]["model_a_test"]["test"],
        "group_weight": 0.3
    },
    # {
    #     "group_name": "EfficientNetV2M_Single_Best",
    #     "type": "individual",
    #     "model_class": get_model("EfficientNetV2MArcFaceModel"),
    #     "model_paths": [
    #         f"{output_root}/EfficientNetV2MArcFaceModel_2025-07-05_20-40_checkpoint.pth"
    #     ],
    #     "transform_params": cfg_inf["transforms"]["model_a_test"]["test"],
    #     "group_weight": 0.1
    # },
    # {
    #     "group_name": "ResNeSt101eModelArcFaceModel",
    #     "type": "individual",
    #     "model_class": get_model("ResNeSt101eModelArcFaceModel"),
    #     "model_paths": [
    #         f"{output_root}/final_model_kfold.pth"
    #     ],
    #     "transform_params": cfg_inf["transforms"]["model_a_test"]["test"],
    #     "group_weight": 0.1
    # },
]
# --------------------------------------------------------------------

def get_group_predictions(models, dataloader, device, tta_transforms):
    """
    모델 '리스트'를 받아, 각 모델에 TTA를 적용하고, 
    그 예측값들의 평균 확률을 반환하는 함수
    """
    for model in models:
        model.eval()

    all_group_avg_probs = []
    with torch.no_grad():
        for images, _, _ in tqdm(dataloader, desc="Inferencing group with TTA"):
            images = images.to(device)
            
            # --- 한 배치에 대한 그룹 내 모든 모델의 TTA 예측을 담을 리스트 ---
            batch_all_models_tta_probs = []

            for model in models:
                # --- 단일 모델에 대한 TTA 예측 ---
                tta_probs_for_one_model = []

                # 원본 이미지 예측
                original_logits = model(images)
                tta_probs_for_one_model.append(F.softmax(original_logits, dim=1))

                # TTA 적용 이미지 예측
                for tta_transform in tta_transforms:
                    augmented_images = tta_transform(images)
                    tta_logits = model(augmented_images)
                    tta_probs_for_one_model.append(F.softmax(tta_logits, dim=1))
                
                # 단일 모델의 TTA 예측값들 평균
                model_tta_avg_probs = torch.stack(tta_probs_for_one_model, dim=0).mean(dim=0)
                batch_all_models_tta_probs.append(model_tta_avg_probs)
            
            # --- 그룹 내 모든 모델의 예측값들 평균 ---
            group_avg_probs = torch.stack(batch_all_models_tta_probs, dim=0).mean(dim=0)
            all_group_avg_probs.append(group_avg_probs)
                
    return torch.cat(all_group_avg_probs, dim=0)

# --- 2. 그룹별 추론 실행 및 결과 취합 ---
all_group_probs = []
test_df = pd.read_csv(test_csv_path)

# TTA에 사용할 변환 리스트
tta_transforms = [
    T.RandomHorizontalFlip(p=1.0),
    T.RandomAffine(degrees=10, scale=(0.9, 1.1)),
]

no_weight = True

# 전체 그룹의 개수를 계산합니다.
num_groups = len(ENSEMBLE_GROUPS)

# 모든 그룹에 할당할 동일한 가중치를 계산합니다.
equal_weight = 1.0 / num_groups

for group_config in ENSEMBLE_GROUPS:
    group_name = group_config['group_name']
    print(f"\n--- 그룹 '{group_name}' 예측 시작 ---")
    if no_weight:
        group_config['group_weight'] = equal_weight
    

    # 1. 그룹에 속한 모델들을 로드
    models_in_group = []
    if group_config['type'] == 'kfold':
        for fold_num in range(1, group_config['n_splits'] + 1):
            path = f"{output_root}/{group_config['experiment_name']}_fold_{fold_num}_checkpoint.pth"
            if not os.path.exists(path): continue
            model = group_config['model_class'](num_classes).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            models_in_group.append(model)
    elif group_config['type'] == 'individual':
        for path in group_config['model_paths']:
            if not os.path.exists(path): continue
            model = group_config['model_class'](num_classes).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            models_in_group.append(model)
    
    if not models_in_group:
        print(f"경고: 그룹 '{group_name}'에서 로드할 모델을 찾지 못했습니다. 건너뜁니다.")
        continue
    
    print(f"{len(models_in_group)}개의 모델 로드 완료.")

    # 2. 그룹 전용 데이터로더 생성
    transform = build_unified_transforms(group_config["transform_params"])
    dataset = get_dataset(cfg_main['DATASET'])(test_csv_path, test_img_dir, transform)
    # DataLoader 생성 시 고정된 배치 사이즈 사용
    dataloader = DataLoader(dataset, batch_size=INFERENCE_BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # 3. 그룹 예측 실행
    group_probs = get_group_predictions(models_in_group, dataloader, device, tta_transforms)
    all_group_probs.append(group_probs)


# --- 3. 최종 결합 ---
print("\n--- 모든 그룹 예측 완료. 최종 결합 수행 ---")

if not all_group_probs:
    raise ValueError("어떤 그룹도 예측을 생성하지 못했습니다. 설정을 확인하세요.")

# 각 그룹의 예측 결과에 최종 가중치 적용
weighted_probs = []
for i, group_probs in enumerate(all_group_probs):
    weight = ENSEMBLE_GROUPS[i]['group_weight']
    weighted_probs.append(group_probs.to(device) * weight)

# 가중치가 적용된 모든 그룹의 확률을 더함
final_probs = torch.stack(weighted_probs, dim=0).sum(dim=0)

# 최종 클래스 예측
final_predictions = final_probs.argmax(dim=1).cpu().numpy()

# 4. 제출 파일 생성
submission_df = pd.read_csv(test_csv_path)
submission_df['target'] = final_predictions
submission_path = f"{output_root}/submission_final_ensemble.csv"
submission_df.to_csv(submission_path, index=False)
print(f"제출 파일 저장 완료: {submission_path}")