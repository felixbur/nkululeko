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

Nothing here depends on a particular architecture: DomainAdversarialHead
only needs a pooled feature vector of known width from whatever backbone
produces one. DannConfig and DannHeads bundle the shared [MODEL] keys and
the per-column heads, so a model only has to expose its feature vector
and add DannHeads.loss() to its task loss (the aasist, mlp, mlp_reg, cnn
and adm models do, via Model._init_dann()). Several nuisance labels (e.g.
source_db for cross-dataset invariance and language for cross-lingual
invariance) can be used at once: DannHeads builds one head per column on
the same feature vector, and DannHeads.loss() sums their losses.

Setting reverse=False turns this into a plain multitask auxiliary head
(gradients flow normally, no reversal) -- an ablation some studies find
outperforms the adversarial version for certain nuisance factors (Liang
et al. 2026, arXiv:2607.23961), available here via the same class
without a separate implementation.
"""

import dataclasses

import numpy as np
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
    `reverse=True` (the adversarial/DANN mode) or with the gradient left
    as is when `reverse=False` (a plain multitask auxiliary head -- no invariance
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
        # The GRL multiplies the backward gradient by -lambda_, so
        # lambda_ = -1 passes it through unchanged (auxiliary head).
        # `lambda_` therefore only matters when reverse=True.
        self.grl = GradientReversalLayer(lambda_ if reverse else -1.0)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.classifier(self.grl(features))


@dataclasses.dataclass
class DannConfig:
    """The shared [MODEL] dann_* keys (read the same way for every model)."""

    columns: list
    lambda_: float
    weight: float
    reverse: bool

    @classmethod
    def from_util(cls, util) -> "DannConfig":
        columns = util.get_dann_columns()
        if not columns:  # DANN off: don't read the other keys
            return cls(columns=[], lambda_=1.0, weight=1.0, reverse=True)
        return cls(
            columns=columns,
            lambda_=util.get_dann_number("dann_lambda"),
            weight=util.get_dann_number("dann_weight"),
            reverse=util.config_val_bool("MODEL", "dann_reverse", True),
        )


class DannHeads(nn.Module):
    """One DomainAdversarialHead per configured column, plus the mapping
    from raw column values to class indices (built from the training data).

    Use DannHeads.build(), which returns None when DANN is off.
    """

    def __init__(self, df_train, feat_dim, cfg: DannConfig, util):
        super().__init__()
        self.columns = list(cfg.columns)
        self.weight = cfg.weight
        self.label_maps = {}
        if len(set(self.columns)) != len(self.columns):
            util.error(f"MODEL.dann_columns lists a column twice: {self.columns}")
        # A ModuleList, not a ModuleDict: column names such as "speaker.id"
        # are not valid module names. Head i belongs to self.columns[i].
        heads = []
        for col in self.columns:
            if col not in df_train.columns:
                util.error(
                    f"MODEL.dann_columns includes '{col}', which is not a "
                    "column of the training data"
                )
            if df_train[col].isna().any():
                util.error(
                    f"MODEL.dann_columns column '{col}' has missing values "
                    "in the training data"
                )
            values = sorted(df_train[col].unique().tolist(), key=str)
            if len(values) < 2:
                util.error(
                    f"MODEL.dann_columns includes '{col}', but the training "
                    f"data has {len(values)} distinct value(s) for it; DANN "
                    "needs at least 2 to discriminate between"
                )
            self.label_maps[col] = {v: i for i, v in enumerate(values)}
            heads.append(
                DomainAdversarialHead(
                    feat_dim=feat_dim,
                    num_classes=len(values),
                    reverse=cfg.reverse,
                    lambda_=cfg.lambda_,
                )
            )
        self.heads = nn.ModuleList(heads)
        util.debug(
            f"DANN heads for {self.columns} (reverse={cfg.reverse}, "
            f"lambda={cfg.lambda_}, weight={cfg.weight})"
        )

    @classmethod
    def build(cls, df_train, feat_dim, cfg: DannConfig, util, device="cpu"):
        """Return DannHeads on `device`, or None if cfg.columns is empty
        (DANN off). The heads must live on the same device as the model's
        features."""
        if not cfg.columns:
            return None
        return cls(df_train, feat_dim, cfg, util).to(device)

    def head(self, col) -> DomainAdversarialHead:
        """The head for column `col`."""
        return self.heads[self.columns.index(col)]

    def encode(self, df) -> np.ndarray:
        """Class indices of every row, shape (len(df), len(columns))."""
        return np.stack(
            [df[col].map(self.label_maps[col]).to_numpy() for col in self.columns],
            axis=1,
        ).astype(np.int64)

    def loss(self, features, domain_labels):
        """Sum of the weighted domain cross-entropies, one per column.

        `domain_labels` is a (batch, len(columns)) long tensor from encode().
        """
        total = 0.0
        for i, head in enumerate(self.heads):
            total = total + self.weight * nn.functional.cross_entropy(
                head(features), domain_labels[:, i]
            )
        return total
