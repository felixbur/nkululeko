# loss_focal.py
"""
Focal Loss for addressing class imbalance and hard example mining.

Focal Loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

Key properties:
- Down-weights loss for well-classified examples (high p_t)
- Focuses training on hard, misclassified examples (low p_t)
- γ (gamma): focusing parameter (higher = more focus on hard examples)
  - γ=0: equivalent to standard cross-entropy
  - γ=2: recommended default
- α (alpha): class weighting factor to handle imbalance
  - α=0.25 for positive class recommended for imbalanced datasets
  
Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    
    Args:
        alpha (float): Weighting factor for the positive class. Default: 0.25
        gamma (float): Focusing parameter. Higher values focus more on hard examples. Default: 2.0
        reduction (str): Specifies reduction to apply to output: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model (before sigmoid), shape (N,) or (N, 1)
            targets: Ground truth labels, shape (N,) or (N, 1), values in {0, 1}
        
        Returns:
            Focal loss value
        """
        # Ensure inputs and targets have the same shape
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)
        
        # Compute cross-entropy: -log(p_t) where p_t is the probability of the true class
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute p_t: probability of the true class
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        
        # Compute focal loss
        focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

