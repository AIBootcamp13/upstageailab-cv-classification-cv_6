import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        
        if alpha is not None:
            # alpha가 텐서가 아니면 텐서로 변환
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            
            # 1. 가중치에 제곱근을 적용하여 부드럽게 만듭니다.
            softened_alpha = torch.sqrt(alpha)
            
            # 2. 부드러워진 가중치를 다시 정규화하여 합이 1이 되도록 합니다.
            self.alpha = softened_alpha / softened_alpha.sum()
        else:
            self.alpha = None
            
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # alpha 텐서를 inputs와 동일한 device로 이동 (안정성 강화)
        if self.alpha is not None and self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        # 이 부분은 기존 코드와 동일합니다.
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            # __init__에서 이미 처리된 alpha 값을 사용
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
}

def get_loss(name: str, params: dict) -> nn.Module:
    return LOSS_REGISTRY[name](**params)