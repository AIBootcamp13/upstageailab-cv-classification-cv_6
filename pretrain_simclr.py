import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from datasets import get_dataset
from utils.utils import *
from models.simclr_model import SimCLRModel
from utils.optimizer_factory import get_optimizer

SEED = 42
set_seed(SEED)


class SimCLRTransform:
    """
    하나의 이미지를 입력받아, 서로 다른 증강이 적용된 두 개의 'view'를 반환합니다.
    """
    def __init__(self, size):
        self.transform = T.Compose([
            # 크기와 위치를 무작위로 변경
            T.RandomResizedCrop(size=size),
            # 50% 확률로 좌우 반전
            T.RandomHorizontalFlip(p=0.5),
            # 80% 확률로 색상 왜곡 적용
            T.RandomApply([
                T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            ], p=0.8),
            # 20% 확률로 흑백으로 변경
            T.RandomGrayscale(p=0.2),
            # 가우시안 블러 적용
            T.GaussianBlur(kernel_size=int(0.1 * size)),
            # 텐서로 변환 및 정규화
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        # 동일한 원본 x에 대해 두 번의 다른 증강을 적용
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


def info_nce_loss(features, temperature=0.1):
    # features: (2 * batch_size, projection_dim) 형태
    batch_size = features.shape[0] // 2
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # 자기 자신과의 비교는 제외
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    
    # positives: 쌍둥이 이미지 간의 유사도
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    # negatives: 다른 모든 이미지와의 유사도
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return F.cross_entropy(logits, labels)

# --- 학습 루프 ---
# ... (모델, 데이터로더, 옵티마이저 설정) ...
# 데이터로더는 레이블 없이 (view1, view2)만 반환해야 함

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


DatasetClass = get_dataset("FastImageDataset")
simclr_transform = SimCLRTransform(size=384)

train_dataset = DatasetClass(
    f"{data_path}/labels_updated.csv",
    f"{data_path}/train/",
    transform=simclr_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=54,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
    worker_init_fn=seed_worker,
    generator=torch.Generator().manual_seed(SEED)
)

model = SimCLRModel().to(device)

optimizer = get_optimizer("AdamW", model.parameters(), {
    "lr": 0.001,
    "weight_decay": 0.01,
})

model.train()
num_epochs = 100
for epoch in range(num_epochs):
    for (view1, view2), _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        images = torch.cat([view1, view2], dim=0)
        
        optimizer.zero_grad()
        
        projections = model(images)
        loss = info_nce_loss(projections)
        
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# 학습이 끝난 모델의 '백본' 가중치만 저장
torch.save(model.backbone.state_dict(), "./output/simclr_pretrained_backbone.pth")