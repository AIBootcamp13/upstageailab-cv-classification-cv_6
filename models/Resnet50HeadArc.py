__all__ = ['ResNet50ArcFaceModel']

import timm
import torch
from torch import nn
import torch.nn.functional as F

from models.ArcMarginProduct import ArcMarginProduct

class ResNet50ArcFaceModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, embedding_size=512, s=30.0, m=0.55):
        """
        ResNet50 백본과 ArcFace 헤드를 사용하는 모델입니다.

        :param num_classes: 최종 분류할 클래스의 수
        :param pretrained: ImageNet으로 사전 학습된 가중치를 사용할지 여부
        :param embedding_size: Neck을 통과한 후의 임베딩 벡터 크기
        """
        super(ResNet50ArcFaceModel, self).__init__()
        
        # 1. Backbone: ResNet50
        self.backbone = timm.create_model('resnet50', pretrained=pretrained, num_classes=0)
        
        # ResNet50의 마지막 레이어 출력 차원은 2048입니다.
        backbone_output_features = self.backbone.num_features # 2048
        
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
            s=s,  # ArcFace의 scale 파라미터 (조정 가능)
            m=m   # ArcFace의 margin 파라미터 (조정 가능)
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