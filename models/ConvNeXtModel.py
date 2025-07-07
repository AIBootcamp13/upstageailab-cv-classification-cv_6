__all__ = ['ConvNeXtModel']

import timm
from torch import nn

class ConvNeXtModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ConvNeXtModel, self).__init__()
        
        # 백본: ConvNeXt
        self.backbone = timm.create_model('convnext_base.fb_in22k', pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features
        
        # 헤드: 분류기
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, labels=None):
        features = self.backbone(x)
        output = self.head(features)
        return output