# loss_pcc.py

import torch


class PearsonCorCoeff(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.mean
        self.sum = torch.sum
        self.sqrt = torch.sqrt

    def forward(self, prediction, ground_truth):
        mean_gt = self.mean(ground_truth, 0)
        mean_pred = self.mean(prediction, 0)
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        cor = self.sum(v_pred * v_gt) / (
            self.sqrt(self.sum(v_pred**2)) * self.sqrt(self.sum(v_gt**2))
        )
        return 1 - cor
