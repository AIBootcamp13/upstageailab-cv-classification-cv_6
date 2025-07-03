__all__ = ['EfficientNetB4Model']

import timm
from torch import nn

class EfficientNetB4Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(EfficientNetB4Model, self).__init__()
        
        # num_classes=0으로 설정하면 timm이 알아서 classifier를 nn.Identity()로 만듭니다.
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
        
        # .num_features 속성으로 피쳐 차원을 간단하게 가져옵니다.
        self.feature_dim = self.backbone.num_features
        
        # 새로운 분류기(head)를 정의합니다.
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        output = self.head(features)
        return output