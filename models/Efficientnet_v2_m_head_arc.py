__all__ = ['EfficientNetV2MArcFaceModel']

import timm
import torch
from torch import nn
import torch.nn.functional as F

from models.ArcMarginProduct import ArcMarginProduct

class EfficientNetV2MArcFaceModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, embedding_size=512, s=30.0, m=0.55):
        super(EfficientNetV2MArcFaceModel, self).__init__()
        
        # 1. Backbone: EfficientNetV2-M
        self.backbone = timm.create_model('tf_efficientnetv2_m.in1k', pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features
        
        # EfficientNetV2-M의 출력 차원은 1280입니다.
        backbone_output_features = 1280
        
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
        features = self.backbone(x)
        embedding = self.neck(features)
        
        if self.training:
            assert labels is not None, "Labels are required during training for ArcFace."
            output = self.head(embedding, labels)
            return output, embedding
        else:
            output = F.linear(F.normalize(embedding), F.normalize(self.head.weight))
            output *= self.head.s
            
            return output