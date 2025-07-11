import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .ArcMarginProduct import ArcMarginProduct


class ConvNeXtV2Model(nn.Module):
    def __init__(self, num_classes=17, model_name='convnextv2_base', pretrained=True, 
                 use_arc_head=True, embedding_dim=512, s=30.0, m=0.55):
        super(ConvNeXtV2Model, self).__init__()
        
        self.use_arc_head = use_arc_head
        self.embedding_dim = embedding_dim
        
        # ConvNeXt V2 백본
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 분류 헤드 제거
            global_pool='avg'
        )
        
        # 백본 출력 차원 확인
        backbone_out_features = self.backbone.num_features
        
        # 넥 (특징 차원 조정)
        self.neck = nn.Sequential(
            nn.Linear(backbone_out_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # 헤드 (분류층)
        if use_arc_head:
            self.head = ArcMarginProduct(embedding_dim, num_classes, s=s, m=m)
        else:
            self.head = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x, labels=None):
        # 백본을 통한 특징 추출
        features = self.backbone(x)
        
        # 넥을 통한 특징 변환
        embeddings = self.neck(features)
        
        # 헤드를 통한 분류
        if self.use_arc_head:
            if self.training and labels is not None:
                output = self.head(embeddings, labels)
            else:
                # 추론 시에는 코사인 유사도 기반으로 로짓 계산
                output = F.linear(F.normalize(embeddings), F.normalize(self.head.weight))
                output *= self.head.s
        else:
            output = self.head(embeddings)
        
        return output
    
    def extract_features(self, x):
        """특징 추출 (앙상블용)"""
        features = self.backbone(x)
        embeddings = self.neck(features)
        return embeddings