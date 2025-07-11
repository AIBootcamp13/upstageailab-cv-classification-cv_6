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
from trainer import *
from trainer.wandb_logger import WandbLogger

# 시드 고정
SEED = 42
set_seed(SEED)

# 설정 로드
cfg = load_config("config/main_config.yaml")

# ConvNeXt-V2 모델로 변경
cfg['MODEL'] = 'ConvNeXtV2Model'
cfg['model_type'] = 'convnext'
cfg['EPOCHS'] = 100  # 테스트용으로 10 epoch
cfg['BATCH_SIZE'] = 4  # 메모리 절약을 위해 배치 크기 더 감소

print(f"=== ConvNeXt-V2 모델 테스트 ===")
print(f"모델: {cfg['MODEL']}")
print(f"에포크: {cfg['EPOCHS']}")

# 나머지 설정은 main.py와 동일
train_transform, val_transform = build_unified_transforms(cfg["transforms"]["train"]), build_unified_transforms(cfg["transforms"]["val"])
DatasetClass = get_dataset(cfg['DATASET'])
ModelClass = get_model(cfg['MODEL'])

# 모델 파라미터 설정
model_params = {
    'num_classes': 17,
    'model_name': 'convnextv2_base',
    'pretrained': True,
    'use_arc_head': True,
    'embedding_dim': 512,
    's': 30.0,
    'm': 0.55
}

# 데이터 로더 설정
data_path = './data'
output_root = './output'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 로드
train_dataset = DatasetClass(
    f"{data_path}/train_valid_set/train-label-fix-v1.csv",
    f"{data_path}/train/",
    transform=train_transform
)
val_dataset = DatasetClass(
    f"{data_path}/train_valid_set/val-v1.csv",
    f"{data_path}/train/",
    transform=val_transform
)

# 데이터 로더 - 메모리 최적화
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg["BATCH_SIZE"],
    shuffle=True,
    num_workers=2,  # 메모리 절약
    pin_memory=False,  # 메모리 절약
    drop_last=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=cfg["BATCH_SIZE"],
    shuffle=False,
    num_workers=2,  # 메모리 절약
    pin_memory=False,  # 메모리 절약
    drop_last=False
)

# 모델 초기화
model = ModelClass(**model_params).to(device)
print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 메모리 최적화 설정
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# CUDA 메모리 분할 크기 설정
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 옵티마이저, 스케줄러, 손실 함수
optimizer = get_optimizer(
    cfg["optimizer"]["name"],
    model.parameters(),
    cfg["optimizer"]["params"]
)
scheduler = get_scheduler(
    cfg["scheduler"]["name"],
    optimizer,
    cfg["scheduler"]["params"]
)
criterion = get_loss(
    cfg["loss"]["name"],
    cfg["loss"]["params"]
)

# 학습 설정
save_path = f'{output_root}/convnext_v2_test.pth'
early_stopping = EarlyStopping(
    patience=cfg["patience"],
    delta=cfg["delta"],
    verbose=True,
    save_path=save_path,
    mode='max'
)

# WandB 로거
logger = WandbLogger(
    project_name="document-type-classification-test",
    run_name=f"convnext_v2_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}",
    config=cfg,
    save_path=save_path
)

# 학습 실행
from trainer.train_loop import training_loop
training_fn = TRAINING_REGISTRY[cfg['training_mode']]

training_args = {}
if cfg["training_mode"] == 'on_amp':
    training_args['scaler'] = GradScaler()

# 그래디언트 누적으로 효과적인 배치 크기 유지
training_args['accumulation_steps'] = cfg.get('accumulation_steps', 8)  # 배치 크기 감소로 인한 누적 증가

# 메모리 최적화를 위한 추가 설정 (training_fn에서 지원하는 파라미터만 사용)

meta_df = pd.read_csv(f"{data_path}/meta_kr.csv")
class_names = meta_df["class_name"].tolist()

print("ConvNeXt-V2 모델 학습 시작...")
try:
    model, valid_max_accuracy = training_loop(
        training_fn,
        model, train_loader, val_loader, train_dataset, val_dataset,
        criterion, optimizer, device, cfg["EPOCHS"],
        early_stopping, logger, class_names, scheduler,
        training_args,
    )
except RuntimeError as e:
    if "out of memory" in str(e):
        print(f"메모리 부족 오류 발생: {e}")
        print("배치 크기를 더 줄이거나 그래디언트 누적을 늘려주세요.")
        torch.cuda.empty_cache()
        raise
    else:
        raise

print(f"ConvNeXt-V2 테스트 완료! 최고 검증 정확도: {valid_max_accuracy:.4f}")
logger.finish()