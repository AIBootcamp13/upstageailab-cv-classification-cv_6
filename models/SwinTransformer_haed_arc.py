__all__ = ['SwinTransformerArcFaceModel']

import timm
import torch
from torch import nn
import torch.nn.functional as F

from models.ArcMarginProduct import ArcMarginProduct


class SwinTransformerArcFaceModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, embedding_size=512, s=30.0, m=0.55):
        super(SwinTransformerArcFaceModel, self).__init__()

        # 1. Backbone: Swin Transformer
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224.ms_in22k', 
            pretrained=pretrained, 
            num_classes=0,
            img_size=(640, 640), # 이미지 사이즈가 224로 고정되어 있어서 이런식으로 바꿔줘야함.
            )
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