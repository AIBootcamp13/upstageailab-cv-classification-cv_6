__all__ = ['ConvNeXtArcFaceModel']

import timm
import torch
from torch import nn
import torch.nn.functional as F

from models.ArcMarginProduct import ArcMarginProduct


class ConvNeXtArcFaceModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, embedding_size=512):
        super(ConvNeXtArcFaceModel, self).__init__()
        
        # 1. Backbone: ConvNeXt
        self.backbone = timm.create_model('convnext_base.fb_in22k', pretrained=pretrained, num_classes=0)
        backbone_output_features = self.backbone.num_features
        
        # 2. Neck: 차원 축소 + 정규화
        self.neck = nn.Sequential(
            nn.Linear(backbone_output_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU()
        )
        
        # 3. Head: ArcFace
        self.head = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            s=30.0,
            m=0.55
        )

    def forward(self, x, labels=None):
        # 백본을 통과하여 특징 추출
        features = self.backbone(x)
        
        # Neck을 통과하여 최종 임베딩 벡터 생성
        embedding = self.neck(features)
        
        # 학습 시에는 레이블을 사용하여 ArcFace 손실 계산
        if self.training and labels is not None:
            output = self.head(embedding, labels)
        # 추론 시에는 코사인 유사도 기반으로 로짓 계산
        else:
            output = F.linear(F.normalize(embedding), F.normalize(self.head.weight))
            output *= self.head.s
        
        return output