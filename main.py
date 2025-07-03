import os
import datetime

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from config.config import load_config
from utils.utils import *
from datasets.transforms import build_unified_transforms

from datasets import get_dataset
from models import get_model

from utils.EarlyStopping import EarlyStopping
from utils.scheduler_factory import get_scheduler
from utils.optimizer_factory import get_optimizer
from trainer.train_loop import training_loop
from trainer.wandb_logger import WandbLogger


# 시드를 고정합니다.
SEED = 42
set_seed(SEED)


cfg = load_config("config/main_config.yaml")
train_transform, val_transform = build_unified_transforms(cfg["transforms"]["train"]), build_unified_transforms(cfg["transforms"]["val"])

DatasetClass = get_dataset(cfg['DATASET'])
ModelClass = get_model(cfg['MODEL'])
cfg_scheduler = cfg["scheduler"]
cfg_optimizer = cfg["optimizer"]


# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data config
data_path = './data'

# output config
output_root = './output'

# training config
num_workers = os.cpu_count() // 2
num_classes = 17
meta_df = pd.read_csv(f"{data_path}/meta_kr.csv")
class_names = meta_df["class_name"].tolist()


date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
filename = f"{cfg['MODEL']}_{date}"

# wandb
logger = WandbLogger(
    project_name="document-type-classification",
    run_name=filename,
    config=cfg,
    save_path=f"{output_root}/checkpoint.pth"
)


# Dataset 정의
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

# DataLoader 정의
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg["BATCH_SIZE"],
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=cfg["BATCH_SIZE"],
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True
)

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=cfg["BATCH_SIZE"],
#     shuffle=True,
#     num_workers=24,
#     pin_memory=True,
#     drop_last=False,
#     persistent_workers=True,
#     prefetch_factor=4,
# )
# val_loader = DataLoader(
#     val_dataset,
#     batch_size=cfg["BATCH_SIZE"],
#     shuffle=True,
#     num_workers=24,
#     pin_memory=True,
#     drop_last=False,
#     persistent_workers=True,
#     prefetch_factor=4,
# )

def unfreeze(model: nn.Module) -> nn.Parameter:
    # 1. 모델의 모든 파라미터를 우선 동결(freeze)합니다.
    for param in model.parameters():
        param.requires_grad = False

    # 2. 특징 추출기(backbone)의 마지막 n개 블록의 동결을 해제(unfreeze)합니다.
    # efficientnet_b3는 7개의 블록(0~6)을 가집니다.
    for i in range(cfg["num_blocks_to_unfreeze"]):
        for param in model.backbone.blocks[-(i+1)].parameters():
            param.requires_grad = True

    # 3. 분류기(head)의 동결을 해제합니다.
    for param in model.head.parameters():
        param.requires_grad = True
        
    # 4. 학습시킬 파라미터만 필터링하여 옵티마이저에 전달합니다.
    # requires_grad=True인 파라미터만 업데이트됩니다.
    params_to_update = filter(lambda p: p.requires_grad, model.parameters())
    
    return params_to_update


def train_block():
    
    # load model
    model: nn.Module = ModelClass(num_classes=num_classes).to(device)
    
    params_to_update = unfreeze(model)
    
    early_stopping = EarlyStopping(patience=cfg["patience"], delta=cfg["delta"], verbose=True, save_path=save_path, mode='max')

    # 손실 함수
    criterion = nn.CrossEntropyLoss()

    # 옵티마이저
    optimizer = get_optimizer(cfg_optimizer["name"], params_to_update, cfg_optimizer["params"])

    # 스케쥴러
    Scheduler = get_scheduler(cfg_scheduler["name"], optimizer, cfg_scheduler['params'])

    model, valid_max_accuracy = training_loop(model, train_loader, val_loader, train_dataset, val_dataset, criterion, optimizer, device, cfg["EPOCHS"], early_stopping, logger, class_names, Scheduler)

    return model, valid_max_accuracy

    
def just_one_train():
    pass


def n_fold_train():
    pass


if __name__ == "__main__":
    
    save_path = f'{output_root}/checkpoint.pth'
    
    # load model
    model: nn.Module = ModelClass(num_classes=num_classes).to(device)
    
    # 1. 모델의 모든 파라미터를 우선 동결(freeze)합니다.
    for param in model.parameters():
        param.requires_grad = False

    # 2. 특징 추출기(backbone)의 마지막 n개 블록의 동결을 해제(unfreeze)합니다.
    # efficientnet_b3는 7개의 블록(0~6)을 가집니다.
    for i in range(cfg["num_blocks_to_unfreeze"]):
        for param in model.backbone.blocks[-(i+1)].parameters():
            param.requires_grad = True

    # 3. 분류기(head)의 동결을 해제합니다.
    for param in model.head.parameters():
        param.requires_grad = True
        
    # 4. 학습시킬 파라미터만 필터링하여 옵티마이저에 전달합니다.
    # requires_grad=True인 파라미터만 업데이트됩니다.
    params_to_update = filter(lambda p: p.requires_grad, model.parameters())

    early_stopping = EarlyStopping(patience=cfg["patience"], delta=cfg["delta"], verbose=True, save_path=save_path, mode='max')

    # 손실 함수
    criterion = nn.CrossEntropyLoss()

    # 옵티마이저
    optimizer = get_optimizer(cfg_optimizer["name"], params_to_update, cfg_optimizer["params"])

    # 스케쥴러
    Scheduler = get_scheduler(cfg_scheduler["name"], optimizer, cfg_scheduler['params'])

    model, valid_max_accuracy = training_loop(model, train_loader, val_loader, train_dataset, val_dataset, criterion, optimizer, device, cfg["EPOCHS"], early_stopping, logger, class_names, Scheduler)