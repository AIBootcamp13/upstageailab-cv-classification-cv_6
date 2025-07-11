__all__ = ['ResNeSt200eModelArcFaceModel']

import timm
import torch
from torch import nn
import torch.nn.functional as F

from models.ArcMarginProduct import ArcMarginProduct  # ArcMarginProduct 클래스를 import

class ResNeSt200eModelArcFaceModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, embedding_size=512, s=30.0, m=0.55):
        super(ResNeSt200eModelArcFaceModel, self).__init__()
        
        # 1. Backbone: ResNeSt101e
        self.backbone = timm.create_model('resnest200e', pretrained=pretrained, num_classes=0)
        
        # 마지막 레이어 출력 차원은 2048입니다. -> 보통 그렇다고 함
        backbone_output_features = self.backbone.num_features # 2048
        
        # 2. Neck: 차원 축소 + 정규화
        self.neck = nn.Sequential(
            nn.Linear(backbone_output_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # 3. Head: ArcFace
        self.head = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            s=s,
            m=m
        )

    def forward(self, x, labels=None):
        # 백본을 통과하여 특징 추출
        features = self.backbone(x)
        
        # Neck을 통과하여 최종 임베딩 벡터 생성
        embedding = self.neck(features)
        
        # 학습 시에는 레이블을 사용하여 ArcFace 손실 계산
        if self.training:
            assert labels is not None, "Labels are required during training for ArcFace."
            output = self.head(embedding, labels)
            return output, embedding
        else:
            output = F.linear(F.normalize(embedding), F.normalize(self.head.weight))
            output *= self.head.s
            
            return output