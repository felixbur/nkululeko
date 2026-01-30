# loss_bce.py
"""
Binary Cross-Entropy loss functions for binary classification.

Includes:
- Standard BCE loss
- Weighted BCE loss for handling class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLoss(nn.Module):
    """
    Standard Binary Cross-Entropy loss with logits.

    Wrapper around PyTorch's BCEWithLogitsLoss for consistency with other loss modules.

    Args:
        reduction (str): Specifies reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
    """

    def __init__(self, reduction="mean"):
        super(BCEWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model (before sigmoid), shape (N,) or (N, 1)
            targets: Ground truth labels, shape (N,) or (N, 1), values in {0, 1}

        Returns:
            BCE loss value
        """
        return self.loss_fn(inputs, targets)


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Weighted BCE loss for handling class imbalance.

    Applies different weights to positive and negative classes to address
    imbalanced datasets where one class is significantly more frequent.

    Args:
        pos_weight (float or Tensor): Weight for positive class.
            - If > 1.0: penalizes false negatives more (focus on recall for positive class)
            - If < 1.0: penalizes false positives more (focus on precision for positive class)
            - Typically set as n_negative / n_positive for balanced error rates
        reduction (str): Specifies reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

    Example:
        # For dataset with 1000 negatives and 200 positives:
        # pos_weight = 1000 / 200 = 5.0
        loss = WeightedBCEWithLogitsLoss(pos_weight=5.0)
    """

    def __init__(self, pos_weight=1.0, reduction="mean"):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.reduction = reduction

        # Convert pos_weight to tensor if it's a scalar
        if isinstance(pos_weight, (int, float)):
            self.pos_weight = torch.tensor([pos_weight])
        else:
            self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model (before sigmoid), shape (N,) or (N, 1)
            targets: Ground truth labels, shape (N,) or (N, 1), values in {0, 1}

        Returns:
            Weighted BCE loss value
        """
        # Move pos_weight to same device as inputs
        pos_weight = self.pos_weight.to(inputs.device)

        return F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=pos_weight, reduction=self.reduction
        )


def compute_class_weights(labels, label_names, mode="balanced"):
    """
    Compute class weights for binary classification.

    Args:
        labels (array-like): Array of labels (0s and 1s or label names)
        label_names (list): List of label names [negative_class, positive_class]
        mode (str): 'balanced' or 'sqrt'
            - 'balanced': weight = n_total / (2 * n_class)
            - 'sqrt': weight = sqrt(n_total / n_class)

    Returns:
        float: pos_weight for WeightedBCEWithLogitsLoss

    Example:
        >>> labels = ['fake', 'fake', 'fake', 'real', 'fake', 'real']
        >>> label_names = ['fake', 'real']
        >>> pos_weight = compute_class_weights(labels, label_names)
        >>> print(pos_weight)  # n_fake / n_real = 4 / 2 = 2.0
    """
    import numpy as np

    # Count classes
    if hasattr(labels, "value_counts"):  # pandas Series
        counts = labels.value_counts()
        n_neg = counts.get(label_names[0], 0)
        n_pos = counts.get(label_names[1], 0)
    else:  # array-like
        labels_array = np.array(labels)
        n_neg = (labels_array == label_names[0]).sum()
        n_pos = (labels_array == label_names[1]).sum()

    if n_pos == 0:
        return 1.0

    if mode == "balanced":
        # Standard balanced weights: n_neg / n_pos
        pos_weight = n_neg / n_pos
    elif mode == "sqrt":
        # Square root balancing (less aggressive)
        pos_weight = np.sqrt(n_neg / n_pos)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'balanced' or 'sqrt'")

    return float(pos_weight)
