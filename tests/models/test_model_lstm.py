"""Unit tests for LSTMModel (nkululeko/models/model_lstm.py).

Same two-level approach as test_model_aasist.py:

1. _FrameSequenceDataset against real (tiny, synthetic) WAV files on disk,
   parametrized over both RNN.feature_source values -- the genuinely new
   integration surface (raw audio I/O, NaT handling, pad/truncate, frame
   computation), independent of the network architecture.
2. LSTMModel.evaluate()/get_probas()/train() with __init__ patched out and
   attributes injected directly, using a trivial stand-in for self.net so
   these tests don't need a real audio pipeline.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
import torch.nn as nn

from nkululeko.models.domain_adversarial import DannConfig, DannHeads
from nkululeko.models.model_lstm import LSTMModel, _FrameSequenceDataset, _RNNNet
from nkululeko.models.rnn_config import RNNConfig

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
    def test_short_and_long_clips_produce_identical_t(self, tmp_path, feature_source):
        """The default DataLoader collate only works if every item has the
        same T -- a short (tiled) clip and a long (truncated) clip must
        produce identical frame counts for a given feature_source."""
        short_wav = tmp_path / "short.wav"
        long_wav = tmp_path / "long.wav"
        _write_wav(short_wav, seconds=0.2)
        _write_wav(long_wav, seconds=2.0)
        cfg = _default_cfg(feature_source=feature_source)

        def _frames_for(wav_path, label):
            index = pd.MultiIndex.from_tuples(
                [(str(wav_path), pd.Timedelta(0), pd.NaT)],
                names=["file", "start", "end"],
            )
            df = pd.DataFrame({"label": [label]}, index=index)
            dataset = _FrameSequenceDataset(df, target="label", cfg=cfg)
            return dataset[0]

        short_frames, short_label = _frames_for(short_wav, 0)
        long_frames, long_label = _frames_for(long_wav, 1)

        assert short_frames.shape[0] == long_frames.shape[0]
        assert short_frames.shape[1] == long_frames.shape[1]
        assert short_label == 0
        assert long_label == 1

    def test_segmented_index_reads_only_the_segment(self, tmp_path, feature_source):
        wav_path = tmp_path / "segmented.wav"
        _write_wav(wav_path, seconds=2.0, sr=16000)
        index = pd.MultiIndex.from_tuples(
            [(str(wav_path), pd.Timedelta(seconds=0.2), pd.Timedelta(seconds=0.4))],
            names=["file", "start", "end"],
        )
        df = pd.DataFrame({"label": [0]}, index=index)
        cfg = _default_cfg(feature_source=feature_source, max_len=3200)

        dataset = _FrameSequenceDataset(df, target="label", cfg=cfg)
        frames, _ = dataset[0]

        assert frames.ndim == 2
        assert frames.shape[0] > 0

    def test_zero_duration_segment_warns_instead_of_crashing(
        self, tmp_path, feature_source
    ):
        wav_path = tmp_path / "clip.wav"
        _write_wav(wav_path, seconds=1.0)
        index = pd.MultiIndex.from_tuples(
            [(str(wav_path), pd.Timedelta(seconds=0.5), pd.Timedelta(seconds=0.5))],
            names=["file", "start", "end"],
        )
        df = pd.DataFrame({"label": [0]}, index=index)
        cfg = _default_cfg(feature_source=feature_source, max_len=1600)
        dataset = _FrameSequenceDataset(df, target="label", cfg=cfg)

        with pytest.warns(UserWarning, match="no audio read"):
            frames, _ = dataset[0]

        assert frames.shape[0] > 0


class TestMelspecShape:
    def test_output_shape_is_t_by_n_mels(self, tmp_path):
        wav_path = tmp_path / "a.wav"
        _write_wav(wav_path, seconds=1.0)
        index = pd.MultiIndex.from_tuples(
            [(str(wav_path), pd.Timedelta(0), pd.NaT)], names=["file", "start", "end"]
        )
        df = pd.DataFrame({"label": [0]}, index=index)
        cfg = _default_cfg(feature_source="melspec", n_mels=10, max_len=16000)
        dataset = _FrameSequenceDataset(df, target="label", cfg=cfg)

        frames, _ = dataset[0]

        assert frames.shape[1] == 10


class TestOpensmileLldShape:
    def test_output_shape_matches_egemaps_lld_dim(self, tmp_path):
        wav_path = tmp_path / "a.wav"
        _write_wav(wav_path, seconds=1.0)
        index = pd.MultiIndex.from_tuples(
            [(str(wav_path), pd.Timedelta(0), pd.NaT)], names=["file", "start", "end"]
        )
        df = pd.DataFrame({"label": [0]}, index=index)
        cfg = _default_cfg(feature_source="opensmile_lld", max_len=16000)
        dataset = _FrameSequenceDataset(df, target="label", cfg=cfg)

        frames, _ = dataset[0]

        assert frames.shape[1] == 25  # eGeMAPSv02 LLD dimensionality


class TestRNNNet:
    def test_forward_shapes(self):
        net = _RNNNet(
            input_size=8,
            hidden_size=4,
            num_layers=2,
            num_classes=3,
            bidirectional=False,
            dropout=0.0,
        )
        x = torch.randn(5, 7, 8)  # (batch, T, input_size)

        logits = net(x)
        assert logits.shape == (5, 3)

        logits, pooled = net(x, return_features=True)
        assert logits.shape == (5, 3)
        assert pooled.shape == (5, 4)
        assert net.feat_dim == 4

    def test_bidirectional_doubles_feat_dim(self):
        net = _RNNNet(
            input_size=8,
            hidden_size=4,
            num_layers=1,
            num_classes=3,
            bidirectional=True,
            dropout=0.0,
        )
        assert net.feat_dim == 8
        _, pooled = net(torch.randn(2, 6, 8), return_features=True)
        assert pooled.shape == (2, 8)


class _TinyNet(nn.Module):
    """Stand-in for _RNNNet: maps a frame sequence straight to logits via
    mean-pooling + a linear layer, independent of the real RNN."""

    feat_dim = 3

    def __init__(self, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x, return_features=False):
        pooled = x.mean(dim=1)
        if pooled.shape[-1] != self.feat_dim:
            pooled = pooled[..., : self.feat_dim]
        logits = self.fc(pooled)
        return (logits, pooled) if return_features else logits


@pytest.fixture
def lstm_model():
    df_train = pd.DataFrame({"label": [0, 1, 2, 0, 1, 2]})
    df_test = pd.DataFrame({"label": [1, 0, 2]})

    with patch.object(LSTMModel, "__init__", return_value=None):
        model = LSTMModel(df_train, df_test, pd.DataFrame(), pd.DataFrame())
        model.target = "label"
        model.class_num = 3
        model.device = "cpu"
        model.run = 0
        model.epoch = 0
        model.df_test = df_test
        model.net = _TinyNet(num_classes=3)
        model.criterion = nn.CrossEntropyLoss()
        model.context = type("Ctx", (), {"labels": [0, 1, 2]})()
        return model


class TestGetLoaderNumWorkers:
    def _model_with_cfg(self, n_jobs):
        with patch.object(LSTMModel, "__init__", return_value=None):
            model = LSTMModel(pd.DataFrame(), pd.DataFrame(), None, None)
            model.cfg = _default_cfg()
            model.target = "label"
            model.n_jobs = n_jobs
            return model

    def _df(self):
        index = pd.MultiIndex.from_tuples(
            [(f"/f{i}.wav", pd.Timedelta(0), pd.NaT) for i in range(4)],
            names=["file", "start", "end"],
        )
        return pd.DataFrame({"label": [0, 1, 0, 1]}, index=index)

    def test_positive_n_jobs_sets_num_workers(self):
        model = self._model_with_cfg(n_jobs=4)
        loader = model.get_loader(self._df(), shuffle=False)
        assert loader.num_workers == 4
        assert loader.persistent_workers is True

    def test_zero_n_jobs_keeps_single_process_loader(self):
        model = self._model_with_cfg(n_jobs=0)
        loader = model.get_loader(self._df(), shuffle=False)
        assert loader.num_workers == 0


class TestTrain:
    def test_train_runs_one_epoch_and_updates_weights(self):
        torch.manual_seed(0)
        with patch.object(LSTMModel, "__init__", return_value=None):
            model = LSTMModel(
                pd.DataFrame({"label": [0, 1, 2, 0]}),
                pd.DataFrame({"label": [1, 0]}),
                pd.DataFrame(),
                pd.DataFrame(),
            )
            model.device = "cpu"
            model.net = _TinyNet(num_classes=3)
            model.dann_heads = None
            model.criterion = nn.CrossEntropyLoss()
            model.optimizer = torch.optim.SGD(model.net.parameters(), lr=0.1)
            model.scheduler = None
            model.scheduler_type = "none"
            model.scheduler_needs_init = False
            model.trainloader = [
                (torch.randn(4, 5, 3), torch.tensor([0, 1, 2, 0])),
                (torch.randn(4, 5, 3), torch.tensor([1, 2, 0, 1])),
            ]
        before = model.net.fc.weight.clone()

        model.train()

        assert torch.isfinite(torch.tensor(model.loss))
        assert not torch.allclose(model.net.fc.weight, before)


def _dann_df():
    index = pd.MultiIndex.from_tuples(
        [(f"/f{i}.wav", pd.Timedelta(0), pd.NaT) for i in range(4)],
        names=["file", "start", "end"],
    )
    return pd.DataFrame(
        {
            "label": [0, 1, 2, 0],
            "source_db": ["a", "a", "b", "b"],
            "language": ["en", "ja", "en", "ja"],
        },
        index=index,
    )


class _Util:
    def debug(self, message):
        pass

    def error(self, message):
        raise RuntimeError(message)


def _dann_model(columns=("source_db",), reverse=True):
    df = _dann_df()
    with patch.object(LSTMModel, "__init__", return_value=None):
        model = LSTMModel(df, df, None, None)
    dann = DannConfig(columns=list(columns), lambda_=1.0, weight=1.0, reverse=reverse)
    model.cfg = _default_cfg(dann=dann)
    model.device = "cpu"
    model.net = _TinyNet(num_classes=3)
    model.target = "label"
    model.n_jobs = 0
    model.criterion = nn.CrossEntropyLoss()
    model.util = _Util()
    model.dann_heads = DannHeads.build(df, model.net.feat_dim, dann, model.util)
    return model, df


class TestDann:
    def test_off_by_default_keeps_two_tuples_and_no_heads(self, tmp_path):
        model, df = _dann_model(columns=())
        wav = tmp_path / "a.wav"
        _write_wav(wav, seconds=0.5)
        df.index = pd.MultiIndex.from_tuples(
            [(str(wav), pd.Timedelta(0), pd.NaT)] * len(df),
            names=["file", "start", "end"],
        )
        assert model.dann_heads is None
        loader = model.get_loader(df, shuffle=False, train=True)
        assert loader.dataset.domain_labels is None
        assert len(loader.dataset[0]) == 2

    def test_one_head_per_column(self):
        model, _ = _dann_model(columns=("source_db", "language"))
        assert model.dann_heads.columns == ["source_db", "language"]
        assert len(model.dann_heads.heads) == 2

    def test_train_loader_yields_domain_labels_test_loader_does_not(self):
        model, df = _dann_model(columns=("source_db", "language"))
        train_ds = model.get_loader(df, shuffle=False, train=True).dataset
        test_ds = model.get_loader(df, shuffle=False).dataset
        assert train_ds.domain_labels.shape == (4, 2)
        assert test_ds.domain_labels is None

    def test_forward_loss_adds_domain_terms(self):
        torch.manual_seed(0)
        model, _ = _dann_model(columns=("source_db", "language"))
        x = torch.randn(4, 5, 3)
        y = torch.tensor([0, 1, 2, 0])
        dom = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        with_dann = model._forward_loss(x, y, dom)
        task_only = model.criterion(model.net(x), y)
        assert with_dann.item() > task_only.item()

    def test_train_updates_heads_and_net(self):
        torch.manual_seed(0)
        model, _ = _dann_model(columns=("source_db",))
        model.optimizer = torch.optim.SGD(
            list(model.net.parameters()) + list(model.dann_heads.parameters()), lr=0.1
        )
        model.scheduler, model.scheduler_type, model.scheduler_needs_init = (
            None,
            "none",
            False,
        )
        model.trainloader = [
            (
                torch.randn(4, 5, 3),
                torch.tensor([0, 1, 2, 0]),
                torch.tensor([[0], [0], [1], [1]]),
            )
        ]
        head = model.dann_heads.head("source_db").classifier[0]
        before = head.weight.clone()
        model.train()
        assert torch.isfinite(torch.tensor(model.loss))
        assert not torch.allclose(head.weight, before)


class TestStoreLoad:
    def _model(self, store_path):
        with patch.object(LSTMModel, "__init__", return_value=None):
            model = LSTMModel(pd.DataFrame(), pd.DataFrame(), None, None)
        model.device = "cpu"
        model.net = _TinyNet(num_classes=3)
        model.store_path = str(store_path)
        model.set_id = lambda run, epoch: None
        model.util = type("U", (), {"error": staticmethod(lambda m: 1 / 0)})()
        return model

    def test_load_restores_stored_weights(self, tmp_path):
        model = self._model(tmp_path / "m.model")
        stored = {k: v.clone() for k, v in model.net.state_dict().items()}
        model.store()
        with torch.no_grad():
            for p in model.net.parameters():
                p.add_(1.0)

        model.load(run=0, epoch=0)

        for k, v in model.net.state_dict().items():
            assert torch.equal(v, stored[k])

    def test_load_missing_file_reports_error(self, tmp_path):
        model = self._model(tmp_path / "missing.model")
        errors = []
        model.util = type("U", (), {"error": staticmethod(errors.append)})()

        model.load(run=0, epoch=0)

        assert errors and "model file not found" in errors[0]


class TestEvaluateAndProbas:
    def test_evaluate_returns_predictions_for_every_row(self, lstm_model):
        # A real DataLoader (not a hand-rolled stand-in): evaluate() slices
        # by loader.batch_size * index, same as model_mlp.py's evaluate(),
        # so it needs a real .dataset/.batch_size pair.
        dataset = torch.utils.data.TensorDataset(
            torch.randn(3, 5, 3), torch.tensor([0, 1, 2])
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        uar, targets, predictions, logits, loss_eval = lstm_model.evaluate(loader)

        assert len(targets) == 3
        assert len(predictions) == 3
        assert logits.shape == (3, 3)
        assert 0.0 <= uar <= 1.0
        assert loss_eval >= 0.0

    def test_get_probas_indexes_by_df_test_and_sums_to_one(self, lstm_model):
        logits = torch.tensor([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.2, 0.2, 0.6]])
        probas = lstm_model.get_probas(logits)

        assert list(probas.index) == list(lstm_model.df_test.index)
        row_sums = probas.sum(axis=1).to_numpy()
        np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0], rtol=1e-5)


def _experiment_context(tmp_path, model_type, target, dann_columns, labels=None):
    import configparser

    from nkululeko.experiment_context import ExperimentContext

    config = configparser.ConfigParser()
    config.read_dict(
        {
            "EXP": {"root": str(tmp_path), "name": "t"},
            "DATA": {"target": target},
            "MODEL": {
                "type": model_type,
                "device": "cpu",
                "n_jobs": "0",
                "dann_columns": dann_columns,
            },
            "RNN": {
                "n_mels": "8",
                "hidden_size": "4",
                "num_layers": "1",
                "max_len": "3200",
                "batch_size": "2",
            },
        }
    )
    return ExperimentContext(config=config, labels=labels)


def _wav_df(tmp_path, target, values):
    paths = []
    for i in range(len(values)):
        path = tmp_path / f"c{i}.wav"
        _write_wav(path, seconds=0.2)
        paths.append((str(path), pd.Timedelta(0), pd.NaT))
    index = pd.MultiIndex.from_tuples(paths, names=["file", "start", "end"])
    return pd.DataFrame(
        {target: values, "source_db": ["a", "b"] * (len(values) // 2)}, index=index
    )


class TestConstructor:
    """Runs the real __init__, which the other tests patch out."""

    @pytest.mark.parametrize("dann_columns", ["[]", "['source_db']"])
    def test_constructed_model_trains(self, tmp_path, dann_columns):
        df = _wav_df(tmp_path, "emotion", [0, 1, 2, 0])
        context = _experiment_context(
            tmp_path, "lstm", "emotion", dann_columns, labels=["a", "b", "c"]
        )

        model = LSTMModel(df, df, None, None, context=context)
        model.train()

        assert (model.dann_heads is not None) == (dann_columns != "[]")
        assert np.isfinite(model.loss)

    def test_get_probas_keeps_classes_missing_from_test(self, lstm_model):
        lstm_model.df_test = pd.DataFrame({"label": [1, 2]})
        logits = torch.tensor([[0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])

        probas = lstm_model.get_probas(logits)

        assert list(probas.columns) == [0, 1, 2]
        np.testing.assert_allclose(
            probas[2].to_numpy(), torch.softmax(logits, dim=1)[:, 2].numpy()
        )


class _ConstNet(nn.Module):
    """Returns fixed logits per row, set by the test."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.pos = 0

    def forward(self, x):
        out = self.logits[self.pos : self.pos + len(x)]
        self.pos += len(x)
        return out


def test_eval_loss_weights_batches_by_size(lstm_model):
    """3 rows at batch_size 2: per-row losses are a, a, b, so the loss
    must be (2a + b) / 3, not the batch mean (a + b) / 2."""
    logits = torch.tensor([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    labels = torch.tensor([0, 0, 0])
    lstm_model.net = _ConstNet(logits)
    dataset = torch.utils.data.TensorDataset(torch.zeros(3, 1, 1), labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    _, _, _, _, loss_eval = lstm_model.evaluate(loader)

    per_row = nn.functional.cross_entropy(logits, labels, reduction="none")
    assert loss_eval == pytest.approx(per_row.mean().item(), rel=1e-5)
