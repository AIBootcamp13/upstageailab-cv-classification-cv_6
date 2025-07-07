import os
import datetime

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold # ✅ K-Fold를 위해 추가

from config.config import load_config
from utils.utils import *
from datasets.transforms import build_unified_transforms

from datasets import get_dataset
from models import get_model

from utils.EarlyStopping import EarlyStopping
from utils.scheduler_factory import get_scheduler
from utils.optimizer_factory import get_optimizer
from trainer import *
from trainer.wandb_logger import WandbLogger


# 시드 고정
SEED = 42
set_seed(SEED)

# 기본 설정 로드
cfg = load_config("config/main_config.yaml")
train_transform, val_transform = build_unified_transforms(cfg["transforms"]["train"]), build_unified_transforms(cfg["transforms"]["val"])
DatasetClass = get_dataset(cfg['DATASET'])
ModelClass = get_model(cfg['MODEL'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 경로 및 데이터 정보
data_path = './data'
output_root = './output'
num_workers = os.cpu_count() // 2
num_classes = 17
meta_df = pd.read_csv(f"{data_path}/meta_kr.csv")
class_names = meta_df["class_name"].tolist()


def unfreeze(model: nn.Module, cfg: dict) -> nn.Parameter:
    # (unfreeze 함수 내용은 동일)
    for param in model.parameters():
        param.requires_grad = False
    for i in range(cfg["num_blocks_to_unfreeze"]):
        for param in model.backbone.blocks[-(i+1)].parameters():
            param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True
    return filter(lambda p: p.requires_grad, model.parameters())

# ====================================================================================
# ✅ 2. 'Fold별로' 실행될 함수 정의 (기존 train_block의 발전된 형태)
# ====================================================================================

def run_fold(fold: int, train_idx: np.ndarray, val_idx: np.ndarray, cfg: dict):
    print(f"========== FOLD {fold} / {cfg['n_splits']} ==========")

    # --- Fold별 설정 (Per-Fold Setup) ---
    
    # Fold별 파일명 및 저장 경로 설정
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"{cfg['MODEL']}_fold{fold}_{date}"
    save_path = f"{output_root}/{run_name}_checkpoint.pth"

    # Fold별 로거 초기화
    logger = WandbLogger(
        project_name="document-type-classification-kfold", # 새 프로젝트 이름
        run_name=run_name,
        config=cfg,
        save_path=save_path
    )

    # Fold별 데이터셋 및 데이터로더 생성
    train_fold_dataset = DatasetClass(meta_df.iloc[train_idx], transform=train_transform)
    val_fold_dataset = DatasetClass(meta_df.iloc[val_idx], transform=val_transform)
    
    train_loader = DataLoader(train_fold_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_fold_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=num_workers, pin_memory=True)

    # Fold별 모델, 옵티마이저, 스케줄러, EarlyStopping 등 초기화
    model: nn.Module = ModelClass(num_classes=num_classes).to(device)
    
    if cfg["use_unfreeze"]:
        params_to_update = unfreeze(model, cfg)
    else:
        params_to_update = model.parameters()

    early_stopping = EarlyStopping(patience=cfg["patience"], delta=cfg["delta"], verbose=True, save_path=save_path, mode='max')
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(cfg["optimizer"]["name"], params_to_update, cfg["optimizer"]["params"])
    scheduler = get_scheduler(cfg["scheduler"]["name"], optimizer, cfg["scheduler"]["params"])

    # Fold별 학습 함수 및 인자 준비
    training_fn = TRAINING_REGISTRY[cfg['training_mode']]
    training_args = {}
    if cfg["training_mode"] == 'on_amp':
        training_args['scaler'] = GradScaler()

    # --- 학습 루프 실행 ---
    model, valid_max_accuracy = training_loop(
        training_fn=training_fn,
        model=model, 
        train_dataloader=train_loader, 
        valid_dataloader=val_loader, 
        train_dataset=train_fold_dataset, 
        val_dataset=val_fold_dataset, 
        criterion=criterion, 
        optimizer=optimizer, 
        device=device, 
        num_epochs=cfg["EPOCHS"], 
        early_stopping=early_stopping, 
        logger=logger, 
        class_names=class_names, 
        scheduler=scheduler,
        training_args=training_args,
    )
    
    return valid_max_accuracy

# ====================================================================================
# ✅ 3. 메인 실행 블록
# ====================================================================================

if __name__ == "__main__":
    
    # Stratified K-Fold 설정
    skf = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=SEED)
    
    all_fold_accuracies = []

    # K-Fold 루프 실행
    for fold, (train_idx, val_idx) in enumerate(skf.split(meta_df, meta_df['class_id'])):
        fold_accuracy = run_fold(fold, train_idx, val_idx, cfg)
        all_fold_accuracies.append(fold_accuracy)

    # 최종 결과 출력
    print("\n========== K-Fold Training Finished ==========")
    for i, acc in enumerate(all_fold_accuracies):
        print(f"Fold {i} Best Accuracy: {acc:.4f}")
    print(f"Average Best Accuracy: {np.mean(all_fold_accuracies):.4f}")