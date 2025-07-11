__all__ = ['ConvNeXtArcFaceSingleModel']

import timm
import torch
from torch import nn
import torch.nn.functional as F

from models.ArcMarginProduct import ArcMarginProduct


class ConvNeXtArcFaceSingleModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, embedding_size=512, s=38.0, m=0.45):
        super(ConvNeXtArcFaceSingleModel, self).__init__()
        
        # 1. Backbone: ConvNeXt
        self.backbone = timm.create_model('convnext_base.fb_in22k', pretrained=pretrained, num_classes=0)
        backbone_output_features = self.backbone.num_features
        
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
        embedding = F.normalize(embedding, p=2, dim=1)
        
        if self.training:
            assert labels is not None, "Labels are required during training for ArcFace."
            output = self.head(embedding, labels)
            return embedding, output
        else:
            output = F.linear(F.normalize(embedding), F.normalize(self.head.weight))
            output *= self.head.s
            
            return output