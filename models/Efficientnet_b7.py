__all__ = ['EfficientNetB7Model']

import timm
from torch import nn

class EfficientNetB7Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(EfficientNetB7Model, self).__init__()
        
        # b7의 적정 입력 이미지 사이즈 600 by 600 이라고 한다.
        self.backbone = timm.create_model('efficientnet_b7', pretrained=pretrained, num_classes=0)
        
        # .num_features 속성으로 피쳐 차원을 간단하게 가져옵니다.
        self.feature_dim = self.backbone.num_features
        
        # 새로운 분류기(head)를 정의합니다.
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        output = self.head(features)
        return output