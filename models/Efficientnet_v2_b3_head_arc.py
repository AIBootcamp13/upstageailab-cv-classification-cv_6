__all__ = ['EfficientNetV2B3ArcFaceModel']

import timm
import torch
from torch import nn
import torch.nn.functional as F

from models.ArcMarginProduct import ArcMarginProduct # ArcMarginProduct 클래스를 파일에서 import

class EfficientNetV2B3ArcFaceModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, embedding_size=512, s=30.0, m=0.55):
        super(EfficientNetV2B3ArcFaceModel, self).__init__()
        
        self.backbone = timm.create_model('tf_efficientnetv2_b3.in1k', pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features
        
        # EfficientNet-B3의 출력 특징 차원은 1536입니다.
        backbone_output_features = 1536 
        
        # 2. Neck: Backbone의 출력을 ArcFace 헤드로 전달하기 전 차원을 축소하고 정규화합니다.
        self.neck = nn.Sequential(
            nn.Linear(backbone_output_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU()
        )
        
        # 3. Head: ArcFace 헤드
        self.head = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            s=s,  # 하이퍼파라미터, 조절 가능
            m=m   # 하이퍼파라미터, 조절 가능
        )

    def forward(self, x, labels=None):
        # 1. Backbone을 통해 특징 추출
        features = self.backbone(x)
        
        # 2. Neck을 통과시켜 최종 임베딩 생성
        embedding = self.neck(features)

        # 3. Head를 통해 로짓 계산
        if self.training:
            assert labels is not None, "Labels are required during training for ArcFace."
            output = self.head(embedding, labels)
            return output, embedding
        else:
            output = F.linear(F.normalize(embedding), F.normalize(self.head.weight))
            output *= self.head.s
            
            return output