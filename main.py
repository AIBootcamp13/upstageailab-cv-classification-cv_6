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


# 시드를 고정합니다.
SEED = 42
set_seed(SEED)


cfg = load_config("config/main_config.yaml")
train_transform, val_transform = build_unified_transforms(cfg["transforms"]["train"]), build_unified_transforms(cfg["transforms"]["val"])

DatasetClass = get_dataset(cfg['DATASET'])
ModelClass = get_model(cfg['MODEL'])
cfg_scheduler = cfg["scheduler"]
cfg_optimizer = cfg["optimizer"]
cfg_loss = cfg["loss"]

training_fn = TRAINING_REGISTRY[cfg['training_mode']]


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

# sampler
sampler = setting_sampler(f"{data_path}/train_valid_set/train-label-fix-v1.csv")

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
    # shuffle=True,
    sampler=sampler,
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
    
    if cfg["use_unfreeze"]:
        params_to_update = unfreeze(model)
    else:
        params_to_update = model.parameters()
    
    early_stopping = EarlyStopping(patience=cfg["patience"], delta=cfg["delta"], verbose=True, save_path=save_path, mode='max')

    # 손실 함수
    criterion = nn.CrossEntropyLoss()

    # 옵티마이저
    optimizer = get_optimizer(cfg_optimizer["name"], params_to_update, cfg_optimizer["params"])

    # 스케쥴러
    Scheduler = get_scheduler(cfg_scheduler["name"], optimizer, cfg_scheduler['params'])

    # amp를 위한 scaler 준비
    training_args = {}
    if cfg["training_mode"] == 'on_amp':
        training_args['scaler'] = GradScaler()

    model, valid_max_accuracy = training_loop(
        training_fn,
        model, train_loader, val_loader, train_dataset, val_dataset, 
        criterion, optimizer, device, cfg["EPOCHS"], 
        early_stopping, logger, class_names, Scheduler,
        training_args,
        )
    return model, valid_max_accuracy

    
def just_one_train():
    pass


def n_fold_train():
    pass


def setup_optimizer_params(
    model: nn.Module, 
    model_type: str, 
    num_layers_to_unfreeze: int,
    backbone_lr: float,
    head_lr: float,
    use_differential_lr: bool,
):
    """
    모델의 동결/해제 상태를 설정하고, 차등 학습률을 적용할 파라미터 그룹을 생성합니다.

    :param model: 설정할 PyTorch 모델 객체
    :param model_type: 모델의 종류 ('resnet', 'efficientnet', 'swin', 'convnext')
    :param num_layers_to_unfreeze: 백본의 마지막에서부터 동결 해제할 레이어(블록)의 수
    :param backbone_lr: 백본에 적용할 낮은 학습률
    :param head_lr: 넥/헤드에 적용할 높은 학습률
    :return: 옵티마이저에 전달할 파라미터 그룹 리스트
    """
    # 1. 모델의 모든 파라미터를 우선 동결(freeze)합니다.
    for param in model.parameters():
        param.requires_grad = False

    # 2. 모델 타입에 따라 백본의 마지막 n개 블록을 동결 해제(unfreeze)합니다.
    model_type_lower = model_type.lower()
    backbone = model.backbone
    
    stages_to_unfreeze = []
    if model_type_lower.startswith('resnet'):
        all_stages = [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
        if num_layers_to_unfreeze > len(all_stages):
            num_layers_to_unfreeze = len(all_stages)
        stages_to_unfreeze = all_stages[-num_layers_to_unfreeze:]
    elif model_type_lower.startswith(('efficientnet', 'swin', 'convnext')):
        if model_type_lower.startswith('swin'): all_stages = backbone.layers
        elif model_type_lower.startswith('convnext'): all_stages = backbone.stages
        else: all_stages = backbone.blocks
        num_total_blocks = len(all_stages)
        if num_layers_to_unfreeze > num_total_blocks:
            num_layers_to_unfreeze = num_total_blocks
        stages_to_unfreeze = all_stages[-num_layers_to_unfreeze:]
    else:
        raise ValueError(f"Unsupported model_type: {model_type}.")

    for stage in stages_to_unfreeze:
        for param in stage.parameters():
            param.requires_grad = True

    # 3. 넥(neck)과 헤드(head)의 동결을 해제합니다.
    if hasattr(model, 'neck'):
        for param in model.neck.parameters():
            param.requires_grad = True
    if hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True
            
    # 4. 차등 학습률을 적용할 파라미터 그룹 생성 혹은 단일 그룹 생성
    #    requires_grad=True인 파라미터만 필터링하여 각 그룹에 포함시킵니다.
    if use_differential_lr:
        param_groups = [
            {
                "params": filter(lambda p: p.requires_grad, model.backbone.parameters()),
                "lr": backbone_lr
            },
            {
                "params": filter(lambda p: p.requires_grad, model.neck.parameters()),
                "lr": head_lr
            },
            {
                "params": filter(lambda p: p.requires_grad, model.head.parameters()),
                "lr": head_lr
            }
        ]
        
        print(f"Unfrozen the last {num_layers_to_unfreeze} backbone layers, neck, and head for model type: {model_type}.")
        print(f"Applied differential learning rate: backbone_lr={backbone_lr}, head_lr={head_lr}")
    else:
        # 단일 학습률 그룹 생성 (학습 가능한 모든 파라미터를 하나의 그룹으로 묶음)
        param_groups = filter(lambda p: p.requires_grad, model.parameters())
        print("Differential learning rate disabled. Using a single LR for all trainable parameters.")
        
    return param_groups


if __name__ == "__main__":
    
    save_path = f'{output_root}/checkpoint.pth'
    
    # load model
    model: nn.Module = ModelClass(num_classes=num_classes).to(device)
    
    if cfg["use_unfreeze"]:
        params_to_update = setup_optimizer_params(
                                model=model,
                                model_type=cfg["model_type"], 
                                num_layers_to_unfreeze=cfg["num_blocks_to_unfreeze"],
                                backbone_lr=cfg["backbone_lr"],
                                head_lr=cfg_optimizer["params"]["lr"],
                                use_differential_lr=cfg["use_differential_lr"]
                            )
    else:
        params_to_update = model.parameters()

    early_stopping = EarlyStopping(patience=cfg["patience"], delta=cfg["delta"], verbose=True, save_path=save_path, mode='max')

    # 손실 함수
    criterion = get_loss(cfg_loss["name"], cfg_loss["params"])

    # 옵티마이저
    optimizer = get_optimizer(cfg_optimizer["name"], params_to_update, cfg_optimizer["params"])

    # 스케쥴러
    Scheduler = get_scheduler(cfg_scheduler["name"], optimizer, cfg_scheduler['params'])
    
    # amp를 위한 scaler 준비
    training_args = {}
    if cfg["training_mode"] == 'on_amp':
        training_args['scaler'] = GradScaler()

    model, valid_max_accuracy = training_loop(
        training_fn,
        model, train_loader, val_loader, train_dataset, val_dataset, 
        criterion, optimizer, device, cfg["EPOCHS"], 
        early_stopping, logger, class_names, Scheduler,
        training_args,
        )