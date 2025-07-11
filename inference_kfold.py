import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from config.config import load_config
from models import get_model
from datasets import get_dataset
from datasets.transforms import build_unified_transforms

# --- 1. 기본 설정 및 경로 ---
# 학습 때 사용했던 config 파일을 그대로 사용합니다.
cfg = load_config("config/main_config.yaml")
ModelClass = get_model(cfg['MODEL'])
DatasetClass = get_dataset(cfg['DATASET'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = './data'
output_root = './output'
num_classes = 17
num_workers = os.cpu_count() // 2

# --- 2. 앙상블할 모델 그룹 이름 설정 (가장 중요) ---
# 학습 스크립트 실행 시 생성되었던 그룹 이름을 여기에 붙여넣으세요.
# 예: "ConvNeXtArcFaceModel_2025-07-09_07-00"
EXPERIMENT_GROUP_NAME = "여기에_학습시_생성된_그룹이름을_입력하세요"
EXPERIMENT_GROUP_NAME = "ConvNeXtArcFaceModel_2025-07-09_09-01"


# --- 3. 모델 로드 ---
ensemble_models = []
print("K-Fold 앙상블 모델 로딩 시작...")

# config 파일에 설정된 n_splits 만큼 모델을 불러옵니다.
for fold_num in range(1, cfg["n_splits"] + 1):
    run_name = f"fold_{fold_num}"
    checkpoint_path = f"{output_root}/{EXPERIMENT_GROUP_NAME}_{run_name}_checkpoint.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"경고: {checkpoint_path}를 찾을 수 없습니다. 건너뜁니다.")
        continue
        
    print(f"Loading: {checkpoint_path}")
    model = ModelClass(num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval() # 평가 모드로 즉시 설정
    ensemble_models.append(model)

if not ensemble_models:
    raise ValueError("로드할 모델이 하나도 없습니다. EXPERIMENT_GROUP_NAME을 확인하세요.")
    
print(f"\n{len(ensemble_models)}개의 모델 로딩 완료.")


# --- 4. 테스트 데이터 준비 ---
# 추론 시에는 학습 때와 달리 데이터 증강이 없는 val_transform을 사용합니다.
test_transform = build_unified_transforms(cfg["transforms"]["test"])
test_dataset = DatasetClass(
    # f"{data_path}/sample_submission.csv",
    f"{data_path}/processed_ensemble_pred.csv",
    # f"{data_path}/test/",
    f"{data_path}/test_modify/",
    transform=test_transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg["BATCH_SIZE"], # config에 설정된 배치 사이즈 사용
    shuffle=False,
    num_workers=num_workers
)


# --- 5. 앙상블 추론 루프 ---
final_predictions = []
with torch.no_grad():
    for images, _, _ in tqdm(test_loader, desc="K-Fold Ensemble Inferencing"):
        images = images.to(device)
        
        batch_probs = []
        # 각 모델로 예측 수행
        for model in ensemble_models:
            # 모델의 forward가 (embedding, logit)을 반환하더라도,
            # model.eval() 상태에서는 logit만 반환하도록 구현했으므로 문제 없습니다.
            logits = model(images) 
            probs = F.softmax(logits, dim=1) # 로짓을 확률로 변환
            batch_probs.append(probs)
        
        # 모든 모델의 예측 확률을 평균냅니다.
        # (Fold 수, 배치 크기, 클래스 수) 형태의 텐서로 쌓은 뒤, 0번 차원에 대해 평균
        avg_probs = torch.stack(batch_probs, dim=0).mean(dim=0)
        
        # 가장 확률이 높은 클래스를 최종 예측으로 선택
        final_class_preds = avg_probs.argmax(dim=1)
        final_predictions.extend(final_class_preds.cpu().numpy())

        
# --- 6. 결과 저장 ---
print("추론 완료. 제출 파일 생성 중...")
submission_df = pd.read_csv(f"{data_path}/sample_submission.csv")
submission_df['target'] = final_predictions
submission_df.to_csv(f"pred_{EXPERIMENT_GROUP_NAME}.csv", index=False)
print(f"제출 파일 저장 완료: pred_{EXPERIMENT_GROUP_NAME}.csv")