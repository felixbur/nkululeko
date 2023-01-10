import torch
from torch.nn.functional import one_hot, softmax

class SoftF1Loss(torch.nn.Module):

    ''' differentiable F1 loss, adapted from
    https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354 
    
    written by Uwe Reichel'''

    def __init__(self, weight=None, epsilon=1e-7, num_classes=-1):
        super().__init__()
        self.weight = weight
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        
        y_true = one_hot(y_true, self.num_classes).to(torch.float32)
        y_pred = softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)

        # unweighted mean F1 loss
        if self.weight is None:
            return 1 - f1.mean()

        # weighted mean F1 loss
        wm = torch.sum(f1 * self.weight) / torch.sum(self.weight)        
        return 1 - wm
