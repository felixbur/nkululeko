"""Unit tests for LSTM_Reg_model (nkululeko/models/model_lstm_regression.py).

Same two-level approach as test_model_lstm.py:

1. _FrameSequenceDataset/_RNNNet directly, checking what differs from the
   classifier: a float label and a 1-output head.
2. LSTM_Reg_model.train()/evaluate()/store()/load() with __init__ patched
   out and attributes injected directly, using a trivial stand-in for
   self.net so these tests don't need a real audio pipeline.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
import torch.nn as nn

from nkululeko.losses.loss_ccc import ConcordanceCorCoeff
from nkululeko.models.domain_adversarial import DannConfig, DannHeads
from nkululeko.models.model_lstm_regression import (
    LSTM_Reg_model,
    _FrameSequenceDataset,
    _RNNNet,
)
from nkululeko.models.rnn_config import RNNConfig
from tests.models.test_model_lstm import _experiment_context, _wav_df

FEATURE_SOURCES = ["melspec", "opensmile_lld"]


def _default_cfg(**overrides):
    fields = {
        "device": "cpu",
        "feature_source": "melspec",
        "n_mels": 8,
        "win_samples": 400,
        "hop_samples": 160,
        "opensmile_set": "eGeMAPSv02",
        "max_len": 16000,
        "hidden_size": 4,
        "num_layers": 1,
        "bidirectional": False,
        "dropout": 0.0,
        "batch_size": 2,
        "dann": DannConfig(columns=[], lambda_=1.0, weight=1.0, reverse=True),
    }
    fields.update(overrides)
    return RNNConfig(**fields)


def _write_wav(path, seconds, sr=16000):
    rng = np.random.default_rng(0)
    signal = rng.uniform(-0.1, 0.1, size=int(seconds * sr)).astype(np.float32)
    sf.write(path, signal, sr)


@pytest.mark.parametrize("feature_source", FEATURE_SOURCES)
class TestFrameSequenceDataset:
    def test_label_is_float_and_t_is_fixed(self, tmp_path, feature_source):
        short_wav = tmp_path / "short.wav"
        long_wav = tmp_path / "long.wav"
        _write_wav(short_wav, seconds=0.2)
        _write_wav(long_wav, seconds=2.0)
        index = pd.MultiIndex.from_tuples(
            [
                (str(short_wav), pd.Timedelta(0), pd.NaT),
                (str(long_wav), pd.Timedelta(0), pd.NaT),
            ],
            names=["file", "start", "end"],
        )
        df = pd.DataFrame({"age": [23, 61.5]}, index=index)
        cfg = _default_cfg(feature_source=feature_source)
        dataset = _FrameSequenceDataset(df, target="age", cfg=cfg)

        short_frames, short_label = dataset[0]
        long_frames, long_label = dataset[1]

        assert short_frames.shape == long_frames.shape
        assert isinstance(short_label, np.float32)
        assert short_label == pytest.approx(23.0)
        assert long_label == pytest.approx(61.5)


class TestRNNNet:
    def test_forward_shapes(self):
        net = _RNNNet(
            input_size=8, hidden_size=4, num_layers=2, bidirectional=False, dropout=0.0
        )
        x = torch.randn(5, 7, 8)

        assert net(x).shape == (5, 1)
        value, pooled = net(x, return_features=True)
        assert value.shape == (5, 1)
        assert pooled.shape == (5, 4)

    def test_bidirectional_doubles_feat_dim(self):
        net = _RNNNet(
            input_size=8, hidden_size=4, num_layers=1, bidirectional=True, dropout=0.0
        )
        assert net.feat_dim == 8


class _TinyNet(nn.Module):
    """Stand-in for _RNNNet: mean-pools the frame sequence and maps it to
    one value via a linear layer."""

    feat_dim = 3

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(self.feat_dim, 1)

    def forward(self, x, return_features=False):
        pooled = x.mean(dim=1)[..., : self.feat_dim]
        value = self.fc(pooled)
        return (value, pooled) if return_features else value


class _Util:
    def __init__(self, measure="mse"):
        self.measure = measure

    def config_val(self, section, key, default):
        if (section, key) == ("MODEL", "measure"):
            return self.measure
        return default

    def debug(self, message):
        pass

    def error(self, message):
        raise RuntimeError(message)


def _model(**attrs):
    with patch.object(LSTM_Reg_model, "__init__", return_value=None):
        model = LSTM_Reg_model(pd.DataFrame(), pd.DataFrame(), None, None)
    model.device = "cpu"
    model.net = _TinyNet()
    model.dann_heads = None
    model.criterion = nn.MSELoss()
    model.util = _Util()
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


def _train_setup(model, batches):
    model.optimizer = torch.optim.SGD(model.net.parameters(), lr=0.1)
    model.scheduler, model.scheduler_type, model.scheduler_needs_init = (
        None,
        "none",
        False,
    )
    model.trainloader = batches


class TestTrain:
    def test_train_runs_one_epoch_and_updates_weights(self):
        torch.manual_seed(0)
        model = _model()
        _train_setup(
            model,
            [
                (torch.randn(4, 5, 3), torch.tensor([1.0, 2.0, 3.0, 4.0])),
                (torch.randn(4, 5, 3), torch.tensor([2.0, 3.0, 4.0, 5.0])),
            ],
        )
        before = model.net.fc.weight.clone()

        model.train()

        assert np.isfinite(model.loss)
        assert not torch.allclose(model.net.fc.weight, before)

    def test_non_finite_batch_is_skipped(self):
        """CCC is undefined on a one-value batch; that batch must neither
        update the weights nor enter the epoch loss."""
        torch.manual_seed(0)
        model = _model(criterion=ConcordanceCorCoeff())
        _train_setup(model, [(torch.randn(1, 5, 3), torch.tensor([3.0]))])
        before = model.net.fc.weight.clone()

        model.train()

        assert model.loss == 0.0
        assert torch.allclose(model.net.fc.weight, before)


class TestEvaluate:
    def _loader(self):
        dataset = torch.utils.data.TensorDataset(
            torch.randn(5, 4, 3), torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        return torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    @pytest.mark.parametrize("measure", ["mse", "mae", "ccc", "pcc"])
    def test_returns_result_for_every_row(self, measure):
        torch.manual_seed(0)
        model = _model(util=_Util(measure=measure))

        result, targets, predictions, loss_eval = model.evaluate(self._loader())

        assert len(targets) == 5
        assert len(predictions) == 5
        assert np.isfinite(result)
        assert loss_eval >= 0.0

    def test_one_value_last_batch_keeps_eval_loss_finite(self):
        """With 1-ccc the last batch of 5 rows at batch_size 2 holds one
        value, for which CCC is undefined; it must not turn the evaluation
        loss into NaN."""
        torch.manual_seed(0)
        model = _model(criterion=ConcordanceCorCoeff(), util=_Util(measure="ccc"))

        _, _, predictions, loss_eval = model.evaluate(self._loader())

        assert np.isfinite(loss_eval)
        assert torch.isfinite(predictions).all()

    def test_unknown_measure_reports_error(self):
        model = _model(util=_Util(measure="nonsense"))
        with pytest.raises(RuntimeError, match="unknown measure"):
            model.evaluate(self._loader())


def _dann_model():
    index = pd.MultiIndex.from_tuples(
        [(f"/f{i}.wav", pd.Timedelta(0), pd.NaT) for i in range(4)],
        names=["file", "start", "end"],
    )
    df = pd.DataFrame(
        {"age": [20.0, 30.0, 40.0, 50.0], "source_db": ["a", "a", "b", "b"]},
        index=index,
    )
    dann = DannConfig(columns=["source_db"], lambda_=1.0, weight=1.0, reverse=True)
    model = _model(cfg=_default_cfg(dann=dann), target="age", n_jobs=0)
    model.dann_heads = DannHeads.build(df, model.net.feat_dim, dann, model.util)
    return model, df


class TestDann:
    def test_train_loader_yields_domain_labels(self):
        model, df = _dann_model()
        dataset = model.get_loader(df, shuffle=False, train=True).dataset
        assert dataset.domain_labels.shape == (4, 1)
        assert model.get_loader(df, shuffle=False).dataset.domain_labels is None

    def test_forward_loss_adds_domain_term(self):
        torch.manual_seed(0)
        model, _ = _dann_model()
        x = torch.randn(4, 5, 3)
        y = torch.tensor([20.0, 30.0, 40.0, 50.0])
        dom = torch.tensor([[0], [0], [1], [1]])
        with_dann = model._forward_loss(x, y, dom)
        task_only = model.criterion(model.net(x).reshape(-1), y)
        assert with_dann.item() > task_only.item()


class TestStoreLoad:
    def test_load_restores_stored_weights(self, tmp_path):
        model = _model(store_path=str(tmp_path / "m.model"))
        model.set_id = lambda run, epoch: None
        stored = {k: v.clone() for k, v in model.net.state_dict().items()}
        model.store()
        with torch.no_grad():
            for p in model.net.parameters():
                p.add_(1.0)

        model.load(run=0, epoch=0)

        for k, v in model.net.state_dict().items():
            assert torch.equal(v, stored[k])

    def test_load_missing_file_reports_error(self, tmp_path):
        model = _model(store_path=str(tmp_path / "missing.model"))
        model.set_id = lambda run, epoch: None
        with pytest.raises(RuntimeError, match="model file not found"):
            model.load(run=0, epoch=0)


class TestConstructor:
    """Runs the real __init__, which the other tests patch out."""

    @pytest.mark.parametrize("dann_columns", ["[]", "['source_db']"])
    def test_constructed_model_trains(self, tmp_path, dann_columns):
        df = _wav_df(tmp_path, "age", [20.0, 30.0, 40.0, 50.0])
        context = _experiment_context(tmp_path, "lstm_reg", "age", dann_columns)

        model = LSTM_Reg_model(df, df, None, None, context=context)
        model.train()

        assert (model.dann_heads is not None) == (dann_columns != "[]")
        assert np.isfinite(model.loss)
