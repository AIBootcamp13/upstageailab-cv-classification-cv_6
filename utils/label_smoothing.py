import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    
    Args:
        smoothing (float): Label smoothing factor (0.0 to 1.0)
        num_classes (int): Number of classes
        ignore_index (int): Index to ignore in loss calculation
    """
    def __init__(self, smoothing=0.1, num_classes=17, ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        """
        Args:
            pred: predictions (N, C) where N is batch size, C is num_classes
            target: ground truth labels (N,)
        """
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
            
            # Handle ignore_index
            if self.ignore_index >= 0:
                true_dist[target == self.ignore_index] = 0.0
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class FocalLossWithLabelSmoothing(nn.Module):
    """
    Focal Loss with Label Smoothing
    
    Args:
        alpha (float): Weighting factor for rare class (default: 1.0)
        gamma (float): Focusing parameter (default: 2.0)
        smoothing (float): Label smoothing factor (default: 0.1)
        num_classes (int): Number of classes
    """
    def __init__(self, alpha=1.0, gamma=2.0, smoothing=0.1, num_classes=17):
        super(FocalLossWithLabelSmoothing, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.num_classes = num_classes
        
    def forward(self, pred, target):
        """
        Args:
            pred: predictions (N, C)
            target: ground truth labels (N,)
        """
        # Apply label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        
        # Calculate focal loss with smoothed labels
        log_pt = F.log_softmax(pred, dim=-1)
        pt = torch.exp(log_pt)
        
        # Apply focal loss formula
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * log_pt
        
        # Apply smoothed labels
        loss = torch.sum(focal_loss * true_dist, dim=-1)
        
        return loss.mean()