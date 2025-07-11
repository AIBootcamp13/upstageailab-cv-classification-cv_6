import os
import datetime
import time

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from pytorch_metric_learning import losses, miners

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


os.environ['TZ'] = 'Asia/Seoul'
time.tzset()


# 기본 설정 로드
cfg = load_config("config/main_config.yaml")
train_transform, val_transform = build_unified_transforms(cfg["transforms"]["train"]), build_unified_transforms(cfg["transforms"]["val"])
# 시드 고정
SEED = 42
set_seed(SEED)
DatasetClass = get_dataset(cfg['DATASET'])
ModelClass = get_model(cfg['MODEL'])
cfg_scheduler = cfg["scheduler"]
cfg_optimizer = cfg["optimizer"]
cfg_loss = cfg["loss"]

training_fn = TRAINING_REGISTRY[cfg['training_mode']]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 경로 및 데이터 정보
data_path = './data'
output_root = './output'
num_workers = os.cpu_count() // 2
num_classes = 17
meta_df = pd.read_csv(f"{data_path}/meta_kr.csv")
class_names = meta_df["class_name"].tolist()


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


def unfreeze(model: nn.Module, cfg: dict) -> nn.Parameter:
    for param in model.parameters():
        param.requires_grad = False
    for i in range(cfg["num_blocks_to_unfreeze"]):
        for param in model.backbone.blocks[-(i+1)].parameters():
            param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True
    return filter(lambda p: p.requires_grad, model.parameters())


def run_fold(fold_num: int, cfg: dict, group_name: str):
    print(f"========== FOLD {fold_num} / {cfg['n_splits']} ==========")

    # --- Fold별 설정 (Per-Fold Setup) ---
    
    # Fold별 파일명 및 저장 경로 설정
    # Fold별 고유 이름과 공통 그룹 이름을 모두 사용
    run_name = f"fold_{fold_num}" # run 이름은 간단하게 Fold 번호만 사용
    save_path = f"{output_root}/{group_name}_{run_name}_checkpoint.pth"

    # Fold별 로거 초기화
    logger = WandbLogger(
        project_name="document-type-classification-kfold",
        run_name=run_name,
        config=cfg,
        group=group_name,
        save_path=save_path,
    )

    # Fold별 데이터셋 및 데이터로더 생성
    train_path = f"{data_path}/train_valid_set/folds/train_fold_{fold_num}_v1.csv"
    val_path = f"{data_path}/train_valid_set/folds/val_fold_{fold_num}_v1.csv"
    
    # sampler
    sampler = setting_sampler(f"{data_path}/train_valid_set/folds/train_fold_{fold_num}.csv")
    
    g = torch.Generator()
    g.manual_seed(cfg["SEED"])
    
    train_fold_dataset = DatasetClass(
        csv=train_path,
        path=f"{data_path}/train/", # 이미지 경로 추가
        transform=train_transform
    )
    val_fold_dataset = DatasetClass(
        csv=val_path,
        path=f"{data_path}/train/", # 이미지 경로 추가
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_fold_dataset,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,  
        )
    val_loader = DataLoader(
        val_fold_dataset, 
        batch_size=cfg["BATCH_SIZE"], 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        )

    # Fold별 모델, 옵티마이저, 스케줄러, EarlyStopping 등 초기화
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
    # Triplet loss
    criterion_triplet = losses.TripletMarginLoss(margin=0.2) # margin 값은 조절 가능
    # miner
    miner = miners.MultiSimilarityMiner()
    # Triplet Loss 가중치
    triplet_loss_weight = cfg["triplet_loss_weight"]
    # 옵티마이저
    optimizer = get_optimizer(cfg_optimizer["name"], params_to_update, cfg_optimizer["params"])
    # 스케쥴러
    scheduler = get_scheduler(cfg_scheduler["name"], optimizer, cfg_scheduler['params'])
    
    # amp를 위한 scaler 준비
    training_args = {}
    if cfg["training_mode"] == 'on_amp':
        training_args['scaler'] = GradScaler()
    else:
        training_args['scaler'] = None

    # training_loop에 전달할 인자에 criterions, miner 추가
    training_args['criterions'] = {'focal': criterion, 'triplet': criterion_triplet}
    training_args['miner'] = miner
    training_args['triplet_loss_weight'] = triplet_loss_weight

    model, valid_max_accuracy = training_loop(
        training_fn,
        model, train_loader, val_loader, train_fold_dataset, val_fold_dataset, 
        optimizer, device, cfg["EPOCHS"], 
        early_stopping, logger, class_names, scheduler,
        training_args,
        )
    
    return valid_max_accuracy


if __name__ == "__main__":
    
    # StratifiedKFold 객체 생성 및 루프를 제거
    all_fold_accuracies = []
    
    # 실험 전체를 대표할 그룹 이름 생성
    experiment_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    experiment_group_name = f"{cfg['MODEL']}_{experiment_date}"

    # 단순 for 루프로 변경
    for fold_num in range(1, cfg["n_splits"] + 1):
        fold_accuracy = run_fold(fold_num, cfg, experiment_group_name) # 수정된 run_fold 호출
        all_fold_accuracies.append(fold_accuracy)

    # 최종 결과 출력
    print("\n========== K-Fold Training Finished ==========")
    for i, acc in enumerate(all_fold_accuracies):
        print(f"Fold {i+1} Best Accuracy: {acc:.4f}")
    print(f"Average Best Accuracy: {np.mean(all_fold_accuracies):.4f}")