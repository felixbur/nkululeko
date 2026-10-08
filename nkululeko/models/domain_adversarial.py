"""Gradient-reversal domain-adversarial training (DANN), model-agnostic.

Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation"
(ICML 2015, arXiv:1409.7495): attach a small classifier head to a
model's pooled feature vector that predicts a *nuisance* label (which
dataset a sample came from, which language it's in, ...); route the
features to that head through a GradientReversalLayer, which passes
activations through unchanged on the forward pass but negates (and
scales) the gradient on the backward pass. Trained jointly with the
main task loss, this pushes the shared representation toward features
the adversarial head *cannot* use to guess the nuisance label -- i.e.
invariant to it -- while the main classification head still trains
normally on top of the same features.

Nothing here depends on AASIST's architecture: DomainAdversarialHead
only needs a pooled feature vector of known width from whatever backbone
produces one (AasistBackend.forward(..., return_features=True)'s
last_hidden; ADM's DeepfakeADMModel could attach the same head to its
own fused representation before its final layer). Two independent heads
with two independent nuisance labels (e.g. source_db for cross-dataset
invariance, language for cross-lingual invariance) can be attached to
the same feature vector simultaneously -- that combination is this
project's "two-axis DANN" -- by constructing two DomainAdversarialHead
instances and summing their losses into the main task loss (see
AasistModel.train()'s DANN branch).

Setting reverse=False turns this into a plain multitask auxiliary head
(gradients flow normally, no reversal) -- an ablation some studies find
outperforms the adversarial version for certain nuisance factors (Liang
et al. 2026, arXiv:2607.23961), available here via the same class
without a separate implementation.
"""

import torch
import torch.nn as nn


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Identity on the forward pass; negates and scales the gradient by
    `lambda_` on the backward pass. `lambda_` is a plain mutable
    attribute (not a buffer/parameter) so a training loop can change it
    without touching the optimizer's state."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return _GradientReversalFunction.apply(x, self.lambda_)


class DomainAdversarialHead(nn.Module):
    """A small MLP classifier over a nuisance label (dataset identity,
    language, ...), fed through a GradientReversalLayer when
    `reverse=True` (the adversarial/DANN mode) or directly when
    `reverse=False` (a plain multitask auxiliary head -- no invariance
    pressure, just an extra supervised signal; see module docstring for
    why this ablation matters)."""

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        reverse: bool = True,
        lambda_: float = 1.0,
    ):
        super().__init__()
        self.reverse = reverse
        self.grl = GradientReversalLayer(lambda_) if reverse else nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    @property
    def lambda_(self) -> float:
        return self.grl.lambda_ if self.reverse else 1.0

    @lambda_.setter
    def lambda_(self, value: float):
        if self.reverse:
            self.grl.lambda_ = value

    def forward(self, features):
        return self.classifier(self.grl(features))
