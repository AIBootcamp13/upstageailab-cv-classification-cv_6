import torch.nn as nn
import timm

class SimCLRModel(nn.Module):
    def __init__(self, backbone_name='convnext_base.fb_in22k', embedding_size=512, projection_dim=128):
        super().__init__()
        # 1. Backbone (기존 모델 사용)
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        backbone_output_features = self.backbone.num_features

        # 2. Projection Head (대조 학습에만 사용)
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_output_features, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, projection_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return projections