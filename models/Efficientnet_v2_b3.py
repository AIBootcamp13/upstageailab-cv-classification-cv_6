__all__ = ['EfficientNetV2B3Model']

import timm
from torch import nn

class EfficientNetV2B3Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(EfficientNetV2B3Model, self).__init__()
        
        # num_classes=0으로 설정하면 timm이 알아서 classifier를 nn.Identity()로 만듭니다.
        self.backbone = timm.create_model('tf_efficientnetv2_b3.in1k', pretrained=pretrained, num_classes=0)
        
        # .num_features 속성으로 피쳐 차원을 간단하게 가져옵니다.
        self.feature_dim = self.backbone.num_features
        
        # 새로운 분류기(head)를 정의합니다.
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim), # 배치 정규화 추가
            nn.Linear(self.feature_dim, 512), # 중간 레이어
            nn.ReLU(),                        # 활성화 함수
            nn.Dropout(0.5),                  # 드롭아웃 (과적합 방지)
            nn.Linear(512, num_classes)       # 최종 출력 레이어
        )

    def forward(self, x, labels=None):
        features = self.backbone(x)
        output = self.head(features)
        return output