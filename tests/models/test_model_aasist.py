"""Unit tests for AasistModel (nkululeko/models/model_aasist.py).

Two levels, matching test_model_adm.py's approach for a model whose real
__init__ needs a full experiment context and downloads a real SSL
checkpoint:

1. _WaveformDataset against real (tiny, synthetic) WAV files on disk --
   the genuinely new integration surface (raw audio I/O, NaT handling,
   pad/truncate), independent of the network architecture.
2. AasistModel.evaluate()/get_probas() with __init__ patched out and
   attributes injected directly (mirroring test_model_adm.py's adm_model
   fixture), using a trivial stand-in for self.net so these tests don't
   need the real AASIST backend or an SSL checkpoint -- that architecture
   is covered separately by test_model_aasist_core.py.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
import torch.nn as nn

from nkululeko.models.aasist_config import AasistConfig
from nkululeko.models.model_aasist import AasistModel, _WaveformDataset


def _default_cfg(**overrides):
    fields = {
        "device": "cpu",
        "ssl_model": "facebook/wav2vec2-xls-r-300m",
        "max_len": 16000,
        "batch_size": 2,
        "ssl_layer_pooling": "last",
        "freeze_ssl_frontend": False,
    }
    fields.update(overrides)
    return AasistConfig(**fields)


def _write_wav(path, seconds, sr=16000):
    rng = np.random.default_rng(0)
    signal = rng.uniform(-0.1, 0.1, size=int(seconds * sr)).astype(np.float32)
    sf.write(path, signal, sr)


class TestWaveformDataset:
    def test_short_clip_is_tiled_to_max_len(self, tmp_path):
        wav_path = tmp_path / "short.wav"
        _write_wav(wav_path, seconds=0.2)  # 3200 samples, shorter than max_len
        index = pd.MultiIndex.from_tuples(
            [(str(wav_path), pd.Timedelta(0), pd.NaT)],
            names=["file", "start", "end"],
        )
        df = pd.DataFrame({"label": [0]}, index=index)

        dataset = _WaveformDataset(df, target="label", cfg=_default_cfg())
        waveform, label = dataset[0]

        assert waveform.shape == (16000,)
        assert label == 0

    def test_long_clip_is_truncated_to_max_len(self, tmp_path):
        wav_path = tmp_path / "long.wav"
        _write_wav(wav_path, seconds=2.0)  # 32000 samples, longer than max_len
        index = pd.MultiIndex.from_tuples(
            [(str(wav_path), pd.Timedelta(0), pd.NaT)],
            names=["file", "start", "end"],
        )
        df = pd.DataFrame({"label": [1]}, index=index)

        dataset = _WaveformDataset(df, target="label", cfg=_default_cfg())
        waveform, label = dataset[0]

        assert waveform.shape == (16000,)
        assert label == 1

    def test_segmented_index_reads_only_the_segment(self, tmp_path):
        wav_path = tmp_path / "segmented.wav"
        _write_wav(wav_path, seconds=2.0, sr=16000)
        # A real segment (0.2s to 0.4s), not a whole-file NaT row.
        index = pd.MultiIndex.from_tuples(
            [(str(wav_path), pd.Timedelta(seconds=0.2), pd.Timedelta(seconds=0.4))],
            names=["file", "start", "end"],
        )
        df = pd.DataFrame({"label": [0]}, index=index)

        dataset = _WaveformDataset(df, target="label", cfg=_default_cfg(max_len=3200))
        waveform, _ = dataset[0]

        # 0.2s at 16kHz = 3200 samples, matching max_len exactly (no tiling).
        assert waveform.shape == (3200,)


class TestWaveformDatasetFormat:
    """The SSL frontend needs 16 kHz mono; other formats must be converted."""

    def _record_padded_signal(self, monkeypatch):
        import nkululeko.models.model_aasist as model_aasist

        seen = {}
        real = model_aasist._pad_or_tile

        def spy(signal, max_len):
            seen["shape"] = signal.shape
            return real(signal, max_len)

        monkeypatch.setattr(model_aasist, "_pad_or_tile", spy)
        return seen

    def test_stereo_48k_is_converted_to_16k_mono(self, tmp_path, monkeypatch):
        rng = np.random.default_rng(0)
        signal = rng.uniform(-0.1, 0.1, size=(2 * 48000, 2)).astype(np.float32)
        wav_path = tmp_path / "stereo48k.wav"
        sf.write(wav_path, signal, 48000)
        index = pd.MultiIndex.from_tuples(
            [(str(wav_path), pd.Timedelta(0), pd.NaT)],
            names=["file", "start", "end"],
        )
        df = pd.DataFrame({"label": [0]}, index=index)
        seen = self._record_padded_signal(monkeypatch)

        dataset = _WaveformDataset(df, target="label", cfg=_default_cfg(max_len=48000))
        waveform, _ = dataset[0]

        assert seen["shape"] == (2 * 16000,)  # 2 s at 16 kHz, one channel
        assert waveform.shape == (48000,)
        assert torch.all(torch.isfinite(waveform))


class _TinyNet(nn.Module):
    """Stand-in for AasistBackend: maps a raw waveform straight to 2
    logits via mean-pooling + a linear layer, so evaluate()/get_probas()
    can be tested without the real SSL frontend or graph-attention stack
    (covered separately by test_model_aasist_core.py)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 2)

    def forward(self, x):
        return self.fc(x.mean(dim=1, keepdim=True))


@pytest.fixture
def aasist_model():
    df_train = pd.DataFrame({"label": [0, 1, 0, 1]})
    df_test = pd.DataFrame({"label": [1, 0]})

    with patch.object(AasistModel, "__init__", return_value=None):
        model = AasistModel(df_train, df_test, pd.DataFrame(), pd.DataFrame())
        model.target = "label"
        model.class_num = 2
        model.device = "cpu"
        model.run = 0
        model.epoch = 0
        model.df_test = df_test
        model.net = _TinyNet()
        model.criterion = nn.CrossEntropyLoss()
        model.context = type("Ctx", (), {"labels": ["real", "fake"]})()
        return model


class TestGetLoaderNumWorkers:
    """get_loader() must parallelize _WaveformDataset's per-item audio I/O
    via MODEL.n_jobs -- a single-process loader (num_workers=0) serializes
    that CPU-bound work with GPU compute. self.n_jobs is set by the base Model class from MODEL.n_jobs
    (default 8); 0 means "no extra workers", matching DataLoader's own
    default and this project's other CPU-light models (e.g. ADM's
    TensorDataset-backed loader, which has no per-item work to
    parallelize).
    """

    def _model_with_cfg(self, n_jobs):
        with patch.object(AasistModel, "__init__", return_value=None):
            model = AasistModel(pd.DataFrame(), pd.DataFrame(), None, None)
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
    """train() runs exactly one epoch over self.trainloader."""

    def test_train_runs_one_epoch_and_updates_weights(self):
        df_train = pd.DataFrame({"label": [0, 1, 0, 1]})
        df_test = pd.DataFrame({"label": [1, 0]})
        with patch.object(AasistModel, "__init__", return_value=None):
            model = AasistModel(df_train, df_test, pd.DataFrame(), pd.DataFrame())
            model.device = "cpu"
            model.net = _TinyNet()
            model.criterion = nn.CrossEntropyLoss()
            model.optimizer = torch.optim.SGD(model.net.parameters(), lr=0.01)
            model.scheduler = None
            model.scheduler_type = "none"
            model.scheduler_needs_init = False
            model.trainloader = [
                (torch.randn(4, 16000), torch.tensor([0, 1, 0, 1])),
                (torch.randn(4, 16000), torch.tensor([1, 0, 1, 0])),
            ]
        before = model.net.fc.weight.clone()

        model.train()

        assert torch.isfinite(torch.tensor(model.loss))
        assert not torch.allclose(model.net.fc.weight, before)


class TestStoreLoad:
    """store()/load() round-trip the backend weights."""

    def _model(self, store_path):
        with patch.object(AasistModel, "__init__", return_value=None):
            model = AasistModel(pd.DataFrame(), pd.DataFrame(), None, None)
        model.device = "cpu"
        model.net = _TinyNet()
        model.store_path = str(store_path)
        # load() calls set_id(run, epoch), which would rebuild store_path
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
    def test_evaluate_returns_predictions_for_every_row(self, aasist_model):
        loader = [
            (torch.zeros(2, 10), torch.tensor([0, 1])),
        ]
        uar, targets, predictions, logits, loss_eval = aasist_model.evaluate(loader)

        assert len(targets) == 2
        assert len(predictions) == 2
        assert logits.shape == (2, 2)
        assert 0.0 <= uar <= 1.0
        assert loss_eval >= 0.0

    def test_get_probas_indexes_by_df_test_and_sums_to_one(self, aasist_model):
        logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        probas = aasist_model.get_probas(logits)

        assert list(probas.index) == list(aasist_model.df_test.index)
        row_sums = probas.sum(axis=1).to_numpy()
        np.testing.assert_allclose(row_sums, [1.0, 1.0], rtol=1e-5)
