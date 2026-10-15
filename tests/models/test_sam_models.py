"""SAM (MODEL.sam) works the same way in every neural model.

Each model's train() goes through Model._optimize(), so with a SAM optimizer
the loss is computed twice per batch (once at the current weights, once at
the perturbed ones) and the weights still train, with or without DANN.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from nkululeko.models.domain_adversarial import DannConfig, DannHeads
from nkululeko.models.model_adm import ADMModel
from nkululeko.models.model_adm_core import DeepfakeADMModel
from nkululeko.models.model_aasist import AasistModel
from nkululeko.models.model_cnn import CNNModel, myCNN
from nkululeko.models.model_mlp import MLPModel
from nkululeko.models.model_mlp_regression import MLP_Reg_model
from nkululeko.optimizers.sam import SAM, is_sam_optimizer

BATCHES = 2


class _Util:
    def debug(self, message):
        pass

    def error(self, message):
        raise RuntimeError(message)

    def config_val(self, section, key, default=None):
        return default


class _Counting(nn.Module):
    """Wraps a criterion and counts how often the loss is computed."""

    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
        self.calls = 0

    def forward(self, *args):
        self.calls += 1
        return self.criterion(*args)


def _optimizer(params, use_sam):
    if use_sam:
        return SAM(params, torch.optim.SGD, rho=0.05, lr=0.1)
    return torch.optim.SGD(params, lr=0.1)


def _dann(model, feat_dim, df):
    cfg = DannConfig(["domain"], 1.0, 1.0, True)
    model.dann_heads = DannHeads.build(df, feat_dim, cfg, model.util)
    return model.dann_heads.encode(df)


def _df(label_col, labels):
    return pd.DataFrame({label_col: labels, "domain": ["a", "a", "b", "b"]})


# Each builder returns (model, net) for a model that trains one epoch of
# BATCHES batches; `net` is the module whose weights must change.


def _mlp(use_sam, with_dann):
    torch.manual_seed(0)
    model = MLPModel.__new__(MLPModel)
    model.util, model.n_jobs, model.target = _Util(), 0, "label"
    model.batch_size, model.device = 2, "cpu"
    model.criterion = _Counting(nn.CrossEntropyLoss())
    model.model = MLPModel.MLP(3, {"a": 8, "b": 4}, 2, False, nn.ReLU())
    df = _df("label", [0, 1, 0, 1])
    domains = _dann(model, model.model.feat_dim, df) if with_dann else None
    params = list(model.model.parameters())
    if with_dann:
        params += list(model.dann_heads.parameters())
    model.optimizer = _optimizer(params, use_sam)
    feats = pd.DataFrame(np.random.rand(4, 3), columns=["f1", "f2", "f3"])
    model.trainloader = model.get_loader(feats, df, False, domains)
    return model, model.model


def _mlp_reg(use_sam, with_dann):
    torch.manual_seed(0)
    model = MLP_Reg_model.__new__(MLP_Reg_model)
    model.util, model.n_jobs, model.target = _Util(), 0, "age"
    model.batch_size, model.device = 2, "cpu"
    model.criterion = _Counting(nn.MSELoss())
    model.model = MLP_Reg_model.MLP(3, {"a": 8, "b": 4}, 1, False, nn.ReLU())
    df = _df("age", [0.1, 0.5, 0.3, 0.9])
    domains = _dann(model, model.model.feat_dim, df) if with_dann else None
    params = list(model.model.parameters())
    if with_dann:
        params += list(model.dann_heads.parameters())
    model.optimizer = _optimizer(params, use_sam)
    feats = pd.DataFrame(np.random.rand(4, 3), columns=["f1", "f2", "f3"])
    model.trainloader = model.get_loader(feats, df, False, domains)
    return model, model.model


def _cnn(use_sam, with_dann):
    torch.manual_seed(0)
    model = CNNModel.__new__(CNNModel)
    model.util, model.device, model.class_num = _Util(), "cpu", 2
    model.criterion = _Counting(nn.CrossEntropyLoss())
    model.model = myCNN([8, 4], 2)
    df = _df("label", [0, 1, 0, 1])
    tensors = [torch.rand(4, 3, 256, 256), torch.tensor([0, 1, 0, 1])]
    params = list(model.model.parameters())
    if with_dann:
        tensors.append(torch.as_tensor(_dann(model, model.model.feat_dim, df)))
        params += list(model.dann_heads.parameters())
    model.optimizer = _optimizer(params, use_sam)
    model.trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*tensors), batch_size=2
    )
    return model, model.model


def _adm(use_sam, with_dann):
    torch.manual_seed(0)
    model = ADMModel.__new__(ADMModel)
    model.util, model.target, model.device = _Util(), "label", "cpu"
    model.batch_size, model.threshold, model.max_grad_norm = 2, 0.5, 0.0
    model.feature_noise = 0.0
    model.ssl_feat_dim, model.fbank_indices, model.stft_indices = 8, [], []
    model.extra_stream_indices = {}
    model.criterion = _Counting(nn.BCEWithLogitsLoss())
    model.model = DeepfakeADMModel(
        ssl_feat_dim=8,
        phase_feat_dim=6,
        fusion="weighted",
        branches=["time", "spectral", "phase"],
        hidden_dim=16,
    )
    df = _df("label", [0, 1, 0, 1])
    domains = _dann(model, model.model.feat_dim, df) if with_dann else None
    params = model._dann_params(model.model)
    model.optimizer = _optimizer(params, use_sam)
    model.scheduler, model.scheduler_type = None, "none"
    model.scheduler_needs_init = False
    feats = pd.DataFrame(np.random.rand(4, 8 + 6))
    # labels are already numeric; build the loader the way the model does
    tensors = [
        torch.tensor(feats.values, dtype=torch.float32),
        torch.tensor(df["label"].values, dtype=torch.float32),
    ]
    if with_dann:
        tensors.append(torch.as_tensor(domains, dtype=torch.long))
    model.trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*tensors), batch_size=2
    )
    return model, model.model


class _AasistNet(nn.Module):
    feat_dim = 4

    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(1, self.feat_dim)
        self.out = nn.Linear(self.feat_dim, 2)

    def forward(self, x, return_features=False):
        feats = self.enc(x.mean(dim=1, keepdim=True))
        logits = self.out(feats)
        return (logits, feats) if return_features else logits


def _aasist(use_sam, with_dann):
    torch.manual_seed(0)
    model = AasistModel.__new__(AasistModel)
    model.util, model.device = _Util(), "cpu"
    model.criterion = _Counting(nn.CrossEntropyLoss())
    model.net = _AasistNet()
    df = _df("label", [0, 1, 0, 1])
    params = list(model.net.parameters())
    domains = None
    if with_dann:
        domains = torch.as_tensor(_dann(model, model.net.feat_dim, df))
        params += list(model.dann_heads.parameters())
    model.optimizer = _optimizer(params, use_sam)
    model.scheduler, model.scheduler_type = None, "none"
    model.scheduler_needs_init = False
    labels = torch.tensor([0, 1, 0, 1])
    batches = []
    for i in range(BATCHES):
        sl = slice(2 * i, 2 * i + 2)
        item = [torch.randn(2, 1600), labels[sl]]
        if with_dann:
            item.append(domains[sl])
        batches.append(tuple(item))
    model.trainloader = batches
    return model, model.net


BUILDERS = {
    "mlp": _mlp,
    "mlp_reg": _mlp_reg,
    "cnn": _cnn,
    "adm": _adm,
    "aasist": _aasist,
}


@pytest.mark.parametrize("with_dann", [False, True], ids=["plain", "dann"])
@pytest.mark.parametrize("name", sorted(BUILDERS))
class TestSamTraining:
    def _first_weight(self, net):
        return next(net.parameters()).detach().clone()

    def test_plain_optimizer_computes_the_loss_once_per_batch(self, name, with_dann):
        model, net = BUILDERS[name](use_sam=False, with_dann=with_dann)
        assert not is_sam_optimizer(model.optimizer)
        model.train()
        assert model.criterion.calls == BATCHES

    def test_sam_computes_the_loss_twice_per_batch(self, name, with_dann):
        model, net = BUILDERS[name](use_sam=True, with_dann=with_dann)
        before = self._first_weight(net)
        model.train()
        assert model.criterion.calls == 2 * BATCHES
        assert np.isfinite(model.loss)
        assert not torch.allclose(next(net.parameters()), before)

    def test_sam_and_plain_take_different_steps(self, name, with_dann):
        sam_model, sam_net = BUILDERS[name](use_sam=True, with_dann=with_dann)
        plain_model, plain_net = BUILDERS[name](use_sam=False, with_dann=with_dann)
        sam_model.train()
        plain_model.train()
        assert not torch.allclose(next(sam_net.parameters()), next(plain_net.parameters()))


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_every_model_hands_its_network_to_optimize(name, monkeypatch):
    """So that SAM can keep the second pass out of BatchNorm statistics."""
    from nkululeko.models.model import Model

    seen = {}
    original = Model._optimize

    def spy(self, *args, **kwargs):
        seen["modules"] = list(kwargs.get("modules", ()))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Model, "_optimize", spy)
    model, net = BUILDERS[name](use_sam=True, with_dann=False)
    model.train()
    assert any(m is net for m in seen["modules"])


def test_aasist_batchnorm_statistics_are_fed_by_the_first_pass_only():
    model, net = _aasist(use_sam=True, with_dann=False)
    net.bn = nn.BatchNorm1d(net.feat_dim)
    net.forward = lambda x, return_features=False: net.out(
        net.bn(net.enc(x.mean(dim=1, keepdim=True)))
    )
    model.optimizer = SAM(net.parameters(), torch.optim.SGD, rho=0.1, lr=0.1)
    momenta = []
    original = model.criterion.forward

    def spy(*args):
        momenta.append(net.bn.momentum)
        return original(*args)

    model.criterion.forward = spy
    model.train()
    assert momenta == [0.1, 0.0] * BATCHES
    assert net.bn.momentum == 0.1


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_sam_also_trains_the_dann_heads(name):
    model, net = BUILDERS[name](use_sam=True, with_dann=True)
    head = model.dann_heads.head("domain").classifier[0]
    before = head.weight.clone()
    model.train()
    assert not torch.allclose(head.weight, before)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_a_batch_skipped_by_sam_is_left_out_of_the_epoch_loss(name):
    """The first batch has a NaN loss: its step is skipped (SAM never applies
    it) and it must not turn the epoch loss into NaN."""
    model, net = BUILDERS[name](use_sam=True, with_dann=False)
    calls = []

    class FirstNaN(nn.Module):
        def __init__(self, criterion):
            super().__init__()
            self.criterion = criterion

        def forward(self, *args):
            calls.append(1)
            loss = self.criterion(*args)
            return loss * float("nan") if len(calls) == 1 else loss

    model.criterion.criterion = FirstNaN(model.criterion.criterion)
    before = next(net.parameters()).detach().clone()
    model.train()
    assert np.isfinite(model.loss)
    assert len(calls) == 3  # one pass for the skipped batch, two for the other
    assert not torch.allclose(next(net.parameters()), before)  # 2nd batch trained
    assert all(torch.isfinite(p).all() for p in net.parameters())


def test_mlp_reg_skips_nan_batches_with_sam():
    model, net = _mlp_reg(use_sam=True, with_dann=False)
    model.criterion = lambda logits, labels: logits.sum() * float("nan")
    before = [p.detach().clone() for p in net.parameters()]
    model.loss = 0.0
    with pytest.warns(RuntimeWarning):  # mean of an empty loss list
        model.train()
    assert all(torch.equal(p, b) for p, b in zip(net.parameters(), before))
