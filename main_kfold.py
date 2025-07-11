import os
import datetime
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from config.config import load_config
from utils.utils import *
from datasets.transforms import build_unified_transforms
from datasets import get_dataset
from models import get_model
from utils.EarlyStopping import EarlyStopping
from utils.scheduler_factory import get_scheduler
from utils.optimizer_factory import get_optimizer
from utils.loss_factory import get_loss
from utils.kfold_training import KFoldTrainer
from trainer import *
from trainer.wandb_logger import WandbLogger

# 시드 고정
SEED = 42
set_seed(SEED)

# 설정 로드
cfg = load_config("config/main_config.yaml")

# 데이터 변환
train_transform, val_transform = build_unified_transforms(cfg["transforms"]["train"]), build_unified_transforms(cfg["transforms"]["val"])

# 클래스 및 설정
DatasetClass = get_dataset(cfg['DATASET'])
ModelClass = get_model(cfg['MODEL'])
cfg_scheduler = cfg["scheduler"]
cfg_optimizer = cfg["optimizer"]
cfg_loss = cfg["loss"]
training_fn = TRAINING_REGISTRY[cfg['training_mode']]

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 경로 설정
data_path = './data'
output_root = './output'

# 데이터 설정
num_workers = os.cpu_count() // 2
num_classes = 17
meta_df = pd.read_csv(f"{data_path}/meta_kr.csv")
class_names = meta_df["class_name"].tolist()

# 파일명 생성
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
filename = f"{cfg['MODEL']}_kfold_{date}"

# 전체 데이터셋 로드
full_dataset = DatasetClass(
    f"{data_path}/train.csv",
    f"{data_path}/train/",
    transform=train_transform
)

def main():
    print("K-Fold Cross Validation 시작")
    print(f"모델: {cfg['MODEL']}")
    print(f"데이터셋 크기: {len(full_dataset)}")
    print(f"클래스 수: {num_classes}")
    
    # K-Fold 트레이너 초기화
    kfold_trainer = KFoldTrainer(n_splits=5, shuffle=True, random_state=SEED)
    
    # 설정 딕셔너리 준비
    training_config = {
        'BATCH_SIZE': cfg['BATCH_SIZE'],
        'EPOCHS': cfg['EPOCHS'],
        'num_classes': num_classes,
        'num_workers': num_workers,
        'patience': cfg['patience'],
        'delta': cfg['delta'],
        'class_names': class_names,
        'training_mode': cfg['training_mode'],
        'optimizer': cfg_optimizer,
        'scheduler': cfg_scheduler,
        'loss': cfg_loss
    }
    
    # K-Fold 교차 검증 수행
    fold_results = kfold_trainer.train_kfold(
        dataset=full_dataset,
        model_class=ModelClass,
        config=training_config,
        training_fn=training_fn,
        device=device
    )
    
    # 결과 저장
    results_path = f"{output_root}/kfold_results_{filename}.json"
    kfold_trainer.save_results(results_path)
    
    # 최고 성능 폴드 찾기
    best_fold = kfold_trainer.get_best_fold('f1_macro')
    print(f"\n최고 성능 폴드: {best_fold['fold'] + 1}")
    print(f"F1 Macro: {best_fold['f1_macro']:.4f}")
    print(f"Accuracy: {best_fold['accuracy']:.4f}")
    
    # 전체 데이터로 최종 모델 학습
    print("\n전체 데이터로 최종 모델 학습 시작...")
    train_final_model(training_config)
    
    print("K-Fold Cross Validation 완료!")

def train_final_model(config):
    """
    K-Fold 결과를 바탕으로 전체 데이터에서 최종 모델 학습
    """
    # 데이터 로더 생성
    train_loader = DataLoader(
        full_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    
    # 검증을 위한 분할 (80:20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config['BATCH_SIZE'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    
    # 모델 초기화
    model = ModelClass(num_classes=num_classes).to(device)
    
    # 옵티마이저, 스케줄러, 손실 함수 설정
    optimizer = get_optimizer(
        config['optimizer']['name'],
        model.parameters(),
        config['optimizer']['params']
    )
    
    scheduler = get_scheduler(
        config['scheduler']['name'],
        optimizer,
        config['scheduler']['params']
    )
    
    criterion = get_loss(
        config['loss']['name'],
        config['loss']['params']
    )
    
    # 조기 종료 설정
    save_path = f'{output_root}/final_model_kfold.pth'
    early_stopping = EarlyStopping(
        patience=config['patience'],
        delta=config['delta'],
        verbose=True,
        save_path=save_path,
        mode='max'
    )
    
    # WandB 로거 설정
    logger = WandbLogger(
        project_name="document-type-classification-kfold",
        run_name=f"final_{filename}",
        config=cfg,
        save_path=save_path
    )
    
    # 학습 인수 설정
    training_args = {}
    if config['training_mode'] == 'on_amp':
        training_args['scaler'] = GradScaler()
    
    # 최종 모델 학습
    from trainer.train_loop import training_loop
    
    final_model, final_accuracy = training_loop(
        training_fn,
        model, train_loader, val_loader, train_subset, val_subset,
        criterion, optimizer, device, config['EPOCHS'],
        early_stopping, logger, config['class_names'], scheduler,
        training_args
    )
    
    print(f"최종 모델 검증 정확도: {final_accuracy:.4f}")
    print(f"최종 모델 저장 경로: {save_path}")
    
    # 로거 종료
    logger.finish()
    
    return final_model, final_accuracy

if __name__ == "__main__":
    main()