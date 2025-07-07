import os

import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as T

from config.config import load_config
from models import get_model
from datasets import get_dataset
from datasets.transforms import build_unified_transforms
from utils.predict_tta import get_tta_predictions


cfg = load_config("config/main_config.yaml")
ModelClass = get_model(cfg['MODEL'])
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

tst_loader = DataLoader(
    tst_dataset,
    batch_size=cfg["BATCH_SIZE"],
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True
)


model = ModelClass(num_classes).to(device)
model.load_state_dict(torch.load(f"{output_root}/checkpoint.pth", map_location="cpu"))

# --- TTA 변환 정의 ---
tta_transforms = [
    T.RandomHorizontalFlip(p=1.0),
    T.RandomRotation(degrees=10),
    T.RandomRotation(degrees=[-10, 10]),
]

# --- 단일 모델 + TTA 추론 루프 ---
final_predictions = []
for images, _, _ in tqdm(tst_loader, desc="Inferencing with TTA"):
    
    # 개선된 TTA 함수를 단일 모델에 바로 적용
    avg_probs = get_tta_predictions(model, images, tta_transforms, device)
    
    # 최종 클래스 예측
    final_class_preds = avg_probs.argmax(dim=1)
    
    final_predictions.extend(final_class_preds.detach().cpu().numpy())



pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
pred_df['target'] = final_predictions

sample_submission_df = pd.read_csv(f"{data_path}/sample_submission.csv")
assert (sample_submission_df['ID'] == pred_df['ID']).all()

pred_df.to_csv("pred.csv", index=False)