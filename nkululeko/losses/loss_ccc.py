# loss_concordance_cor_coeff.py

import torch


class ConcordanceCorCoeff(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.mean
        self.var = torch.var
        self.sum = torch.sum
        self.sqrt = torch.sqrt
        self.std = torch.std

    def forward(self, prediction, ground_truth):
        # eps guards both divisions against exact zero - e.g. a batch where
        # predictions haven't diverged yet (v_pred all ~0, common right
        # after initialization, especially under fp16) otherwise makes
        # cor = 0/0 = NaN. A single NaN loss permanently corrupts the
        # optimizer's running moments, silently ending training for good
        # rather than just harming one step.
        eps = 1e-8
        ground_truth = ground_truth.float()
        mean_gt = self.mean(ground_truth, 0)
        mean_pred = self.mean(prediction, 0)
        var_gt = self.var(ground_truth, 0)
        var_pred = self.var(prediction, 0)
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        cor = self.sum(v_pred * v_gt) / (
            self.sqrt(self.sum(v_pred**2)) * self.sqrt(self.sum(v_gt**2)) + eps
        )
        sd_gt = self.std(ground_truth)
        sd_pred = self.std(prediction)
        numerator = 2 * cor * sd_gt * sd_pred
        denominator = var_gt + var_pred + (mean_gt - mean_pred) ** 2 + eps
        ccc = numerator / denominator
        return 1 - ccc
