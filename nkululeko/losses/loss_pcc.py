# loss_pcc.py

import torch


class PearsonCorCoeff(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.mean
        self.sum = torch.sum
        self.sqrt = torch.sqrt

    def forward(self, prediction, ground_truth):
        # eps guards against a batch where predictions haven't diverged yet
        # (v_pred all ~0, common right after initialization, especially
        # under fp16), which otherwise makes cor = 0/0 = NaN. A single NaN
        # loss permanently corrupts the optimizer's running moments, so this
        # silently ends training for good rather than just harming one step.
        eps = 1e-8
        ground_truth = ground_truth.float()
        mean_gt = self.mean(ground_truth, 0)
        mean_pred = self.mean(prediction, 0)
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        cor = self.sum(v_pred * v_gt) / (
            self.sqrt(self.sum(v_pred**2)) * self.sqrt(self.sum(v_gt**2)) + eps
        )
        return 1 - cor
