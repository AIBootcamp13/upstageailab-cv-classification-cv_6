import torch
import torch.nn as nn
import torch.nn.functional as F
from .label_smoothing import LabelSmoothingCrossEntropy, FocalLossWithLabelSmoothing

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # inputs: 모델의 출력 (logits), targets: 정답 레이블
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            # alpha 가중치 적용 (클래스 불균형 처리)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



LOSS_REGISTRY: dict[str, nn.Module] = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "FocalLoss": FocalLoss,
    "LabelSmoothingCrossEntropy": LabelSmoothingCrossEntropy,
    "FocalLossWithLabelSmoothing": FocalLossWithLabelSmoothing,
}

def get_loss(name: str, params: dict) -> nn.Module:
    return LOSS_REGISTRY[name](**params)