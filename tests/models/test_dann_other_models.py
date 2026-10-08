"""DANN (MODEL.dann_columns) wiring in mlp_reg, cnn and adm.

The shared pieces (DannHeads, gradient reversal) are tested in
test_domain_adversarial.py; here we check each model exposes its feature
vector, trains its heads, and still evaluates when the train loader carries
a third (domain-label) item.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from nkululeko.models.domain_adversarial import DannConfig, DannHeads
from nkululeko.models.model_adm import ADMModel
from nkululeko.models.model_adm_core import DeepfakeADMModel
from nkululeko.models.model_cnn import CNNModel, myCNN
from nkululeko.models.model_mlp_regression import MLP_Reg_model


class _Util:
    def debug(self, message):
        pass

    def error(self, message):
        raise RuntimeError(message)

    def config_val(self, section, key, default=None):
        return default


def _heads(feat_dim, n, device="cpu"):
    df = pd.DataFrame({"domain": ["a", "b"] * (n // 2)})
    cfg = DannConfig(columns=["domain"], lambda_=1.0, weight=1.0, reverse=True)
    return DannHeads.build(df, feat_dim, cfg, _Util(), device), df


# ---------------------------------------------------------------- mlp_reg


class TestMlpReg:
    def _model(self, with_dann):
        torch.manual_seed(0)
        feats = pd.DataFrame(np.random.rand(4, 3), columns=["f1", "f2", "f3"])
        df = pd.DataFrame({"age": [0.1, 0.5, 0.3, 0.9], "domain": ["a", "a", "b", "b"]})
        model = MLP_Reg_model.__new__(MLP_Reg_model)
        model.util = _Util()
        model.n_jobs = 0
        model.target = "age"
        model.batch_size = 2
        model.device = "cpu"
        model.criterion = torch.nn.MSELoss()
        model.model = MLP_Reg_model.MLP(
            3, {"a": 8, "b": 4}, 1, False, torch.nn.ReLU()
        )
        params = list(model.model.parameters())
        domain_labels = None
        if with_dann:
            cfg = DannConfig(["domain"], 1.0, 1.0, True)
            model.dann_heads = DannHeads.build(df, model.model.feat_dim, cfg, model.util)
            params += list(model.dann_heads.parameters())
            domain_labels = model.dann_heads.encode(df)
        model.optimizer = torch.optim.SGD(params, lr=0.1)
        model.trainloader = model.get_loader(feats, df, False, domain_labels)
        return model

    def test_returns_hidden_features(self):
        mlp = MLP_Reg_model.MLP(3, {"a": 8, "b": 4}, 1, False, torch.nn.ReLU())
        x = torch.rand(5, 3)
        out, hidden = mlp(x, return_features=True)
        assert hidden.shape == (5, mlp.feat_dim) == (5, 8)
        assert torch.equal(out, mlp(x))

    def test_dann_off_loader_yields_pairs(self):
        model = self._model(with_dann=False)
        assert len(next(iter(model.trainloader))) == 2

    def test_train_with_dann_updates_heads(self):
        model = self._model(with_dann=True)
        head = model.dann_heads.head("domain").classifier[0]
        before = head.weight.clone()
        model.train()
        assert np.isfinite(model.loss)
        assert not torch.allclose(head.weight, before)

    def test_evaluate_accepts_domain_label_batches(self):
        model = self._model(with_dann=True)
        result, targets, predictions = model.evaluate_model(
            model.model, model.trainloader, "cpu"
        )
        assert len(targets) == len(predictions) == 4


# -------------------------------------------------------------------- cnn


class TestCnn:
    def _model(self):
        torch.manual_seed(0)
        model = CNNModel.__new__(CNNModel)
        model.util = _Util()
        model.device = "cpu"
        model.class_num = 2
        model.criterion = torch.nn.CrossEntropyLoss()
        model.model = myCNN([8, 4], 2)
        df = pd.DataFrame({"domain": ["a", "b", "a", "b"]})
        cfg = DannConfig(["domain"], 1.0, 1.0, True)
        model.dann_heads = DannHeads.build(df, model.model.feat_dim, cfg, model.util)
        model.optimizer = torch.optim.SGD(
            list(model.model.parameters()) + list(model.dann_heads.parameters()),
            lr=0.1,
        )
        images = torch.rand(4, 3, 256, 256)
        labels = torch.tensor([0, 1, 0, 1])
        domains = torch.as_tensor(model.dann_heads.encode(df))
        model.trainloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(images, labels, domains), batch_size=2
        )
        return model

    def test_returns_hidden_features(self):
        net = myCNN([8, 4], 2)
        x = torch.rand(2, 3, 256, 256)
        logits, hidden = net(x, return_features=True)
        assert hidden.shape == (2, net.feat_dim) == (2, 8)  # layers are sorted
        assert torch.equal(logits, net(x))

    def test_train_with_dann_updates_heads(self):
        model = self._model()
        head = model.dann_heads.head("domain").classifier[0]
        before = head.weight.clone()
        model.train()
        assert np.isfinite(model.loss)
        assert not torch.allclose(head.weight, before)

    def test_evaluate_accepts_domain_label_batches(self):
        model = self._model()
        uar, targets, predictions, logits = model.evaluate(
            model.model, model.trainloader, "cpu"
        )
        assert logits.shape == (4, 2)


# -------------------------------------------------------------------- adm


class TestAdm:
    def _core(self, branches=("time", "spectral", "phase"), hidden_dim=16):
        return DeepfakeADMModel(
            ssl_feat_dim=8,
            phase_feat_dim=6,
            fusion="weighted",
            branches=list(branches),
            hidden_dim=hidden_dim,
        )

    @pytest.mark.parametrize(
        "branches", [("time",), ("time", "spectral"), ("time", "spectral", "phase")]
    )
    def test_features_are_concatenated_branch_activations(self, branches):
        core = self._core(branches)
        core.eval()
        args = (torch.rand(3, 8), torch.rand(3, 5), torch.rand(3, 6))
        with torch.no_grad():
            score, feats = core(*args, return_features=True)
            plain = core(*args)
        assert feats.shape == (3, core.feat_dim) == (3, len(branches) * 8)
        assert torch.equal(score, plain)

    def _model(self):
        torch.manual_seed(0)
        model = ADMModel.__new__(ADMModel)
        model.util = _Util()
        model.target = "label"
        model.device = "cpu"
        model.batch_size = 2
        model.threshold = 0.5
        model.max_grad_norm = 0.0
        model.feature_noise = 0.0
        model.ssl_feat_dim = 8
        model.fbank_indices = []
        model.stft_indices = []
        model.extra_stream_indices = {}
        model.criterion = torch.nn.BCEWithLogitsLoss()
        model.model = self._core()
        df = pd.DataFrame({"label": [0, 1, 0, 1], "domain": ["a", "a", "b", "b"]})
        cfg = DannConfig(["domain"], 1.0, 1.0, True)
        model.dann_heads = DannHeads.build(df, model.model.feat_dim, cfg, model.util)
        model.optimizer = torch.optim.SGD(
            list(model.model.parameters()) + list(model.dann_heads.parameters()),
            lr=0.1,
        )
        model.scheduler, model.scheduler_type, model.scheduler_needs_init = (
            None,
            "none",
            False,
        )
        feats = pd.DataFrame(np.random.rand(4, 8 + 6))
        model.trainloader = model.get_loader(
            feats, df, False, model.dann_heads.encode(df)
        )
        return model

    def test_train_with_dann_updates_heads(self):
        model = self._model()
        head = model.dann_heads.head("domain").classifier[0]
        before = head.weight.clone()
        model.train()
        assert np.isfinite(model.loss)
        assert not torch.allclose(head.weight, before)

    def test_grad_clipping_covers_the_dann_heads(self, monkeypatch):
        model = self._model()
        model.max_grad_norm = 1.0
        clipped = []
        monkeypatch.setattr(
            torch.nn.utils,
            "clip_grad_norm_",
            lambda params, max_norm: clipped.extend(params),
        )
        model.train()
        head_params = list(model.dann_heads.parameters())
        assert head_params
        assert all(any(p is c for c in clipped) for p in head_params)
        assert all(any(p is c for c in clipped) for p in model.model.parameters())

    def test_evaluate_accepts_domain_label_batches(self):
        model = self._model()
        uar, targets, predictions, logits = model.evaluate(
            model.model, model.trainloader, "cpu"
        )
        assert len(logits) == 4

    def test_loader_without_domain_labels_yields_pairs(self):
        model = self._model()
        df = pd.DataFrame({"label": [0, 1]})
        loader = model.get_loader(pd.DataFrame(np.random.rand(2, 14)), df, False)
        assert len(next(iter(loader))) == 2
