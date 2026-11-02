"""RawBoost on the fly in AASIST's training loader ([AASIST] rawboost)."""

import configparser

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

import nkululeko.glob_conf as glob_conf
from nkululeko.augmenting.rawboost import RawBoostParams
from nkululeko.models.aasist_config import AasistConfig
from nkululeko.models.domain_adversarial import DannConfig
from nkululeko.models.model_aasist import AasistModel, _rawboost_rng, _WaveformDataset
from nkululeko.utils.errors import NkululukoError
from nkululeko.utils.util import Util

MAX_LEN = 4000


def _cfg(rawboost=None, batch_size=2):
    return AasistConfig(
        device="cpu",
        ssl_model="x",
        max_len=MAX_LEN,
        batch_size=batch_size,
        ssl_layer_pooling="last",
        freeze_ssl_frontend=False,
        dann=DannConfig(columns=[], lambda_=1.0, weight=1.0, reverse=True),
        rawboost=rawboost,
    )


def _df(tmp_path, n=4):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        path = tmp_path / f"{i}.wav"
        sf.write(path, rng.uniform(-0.3, 0.3, MAX_LEN).astype(np.float32), 16000)
        rows.append((str(path), pd.Timedelta(0), pd.NaT))
    index = pd.MultiIndex.from_tuples(rows, names=["file", "start", "end"])
    return pd.DataFrame({"label": [0, 1] * (n // 2)}, index=index)


def _util(tmp_path, aasist=None, augment=None):
    config = configparser.ConfigParser()
    config["EXP"] = {"type": "classification", "name": "x", "root": str(tmp_path)}
    config["DATA"] = {"target": "label", "databases": "['a']"}
    config["MODEL"] = {"type": "aasist"}
    config["FEATS"] = {"type": "[]"}
    config["AASIST"] = aasist or {}
    config["AUGMENT"] = augment or {}
    glob_conf.config = config
    return Util("test")


@pytest.fixture(autouse=True)
def _reset():
    yield
    glob_conf.config = None


class TestConfig:
    def test_off_by_default(self, tmp_path):
        assert AasistConfig.from_util(_util(tmp_path)).rawboost is None

    def test_on_reads_the_augment_settings(self, tmp_path):
        util = _util(
            tmp_path,
            {"rawboost": "True"},
            {"rawboost_algo": "3", "rawboost_snr_min": "5"},
        )
        params = AasistConfig.from_util(util).rawboost
        assert (params.algo, params.snr_min, params.snr_max) == (3, 5, 40)

    def test_settings_are_ignored_while_off(self, tmp_path):
        util = _util(tmp_path, {"rawboost": "False"}, {"rawboost_algo": "0"})
        assert AasistConfig.from_util(util).rawboost is None

    def test_bad_settings_are_an_error(self, tmp_path):
        util = _util(tmp_path, {"rawboost": "True"}, {"rawboost_algo": "0"})
        with pytest.raises(NkululukoError, match="rawboost_algo"):
            AasistConfig.from_util(util)


class TestDataset:
    def test_without_rawboost_an_item_is_always_the_same(self, tmp_path):
        data = _WaveformDataset(_df(tmp_path), "label", _cfg())
        assert torch.equal(data[0][0], data[0][0])

    def test_with_rawboost_every_read_has_new_noise(self, tmp_path):
        df = _df(tmp_path)
        plain = _WaveformDataset(df, "label", _cfg())
        noisy = _WaveformDataset(df, "label", _cfg(), rawboost=RawBoostParams(algo=3))
        first, again = noisy[0][0], noisy[0][0]
        assert first.shape == (MAX_LEN,) and torch.isfinite(first).all()
        assert not torch.allclose(first, plain[0][0])
        assert not torch.allclose(first, again)
        assert noisy[0][1] == plain[0][1]

    def test_workers_draw_different_noise(self, tmp_path):
        """The noise comes from numpy's global random state: every
        dataloader worker must have its own, or all workers add the same
        noise."""
        df = _df(tmp_path, n=8)
        data = _WaveformDataset(df, "label", _cfg(), rawboost=RawBoostParams(algo=3))
        loader = torch.utils.data.DataLoader(
            data, batch_size=1, num_workers=2, persistent_workers=True
        )
        # every row has other audio: remove it, to compare the noise only
        plain = _WaveformDataset(df, "label", _cfg())
        noise = {}
        for epoch in range(2):
            for i, (wave, _) in enumerate(loader):
                noise[(epoch, i)] = (wave[0] - plain[i][0]).numpy()
        keys = list(noise)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                assert not np.allclose(noise[keys[a]], noise[keys[b]]), (
                    keys[a],
                    keys[b],
                )


class TestSeedReproducibility:
    """GH #483: apply_rawboost() draws from numpy's global RNG, which
    MODEL.random_seed only seeded via torch.manual_seed(), not np.random.seed()
    -- so the noise differed between two runs with the same seed. Fixed not
    by seeding the global RNG (which a reviewer noted drifts whenever
    anything else draws from numpy in between, and makes reproducibility
    depend on MODEL.n_jobs/worker layout matching too) but by deriving a
    dedicated RandomState per sample from (MODEL.random_seed, sample index)
    via _rawboost_rng -- deterministic no matter what else runs in the same
    process, and independent of num_workers."""

    def test_same_seed_reproduces_the_same_noise(self, tmp_path):
        df = _df(tmp_path)

        first_run = _WaveformDataset(
            df, "label", _cfg(), rawboost=RawBoostParams(algo=3), rawboost_seed=123
        )
        first = [first_run[i][0].clone() for i in range(len(df))]

        second_run = _WaveformDataset(
            df, "label", _cfg(), rawboost=RawBoostParams(algo=3), rawboost_seed=123
        )
        second = [second_run[i][0].clone() for i in range(len(df))]

        for a, b in zip(first, second):
            assert torch.equal(a, b)

    def test_different_samples_get_different_noise_under_the_same_seed(self, tmp_path):
        df = _df(tmp_path)
        data = _WaveformDataset(
            df, "label", _cfg(), rawboost=RawBoostParams(algo=3), rawboost_seed=123
        )
        assert not torch.allclose(data[0][0], data[1][0])

    def test_reproducible_independent_of_worker_count(self, tmp_path):
        """The actual guarantee this issue asks for: the same MODEL.random_seed
        produces identical RawBoost noise whether MODEL.n_jobs is 0 or > 0,
        unlike the old per-worker np.random.seed(base + worker_id) design."""
        df = _df(tmp_path, n=4)

        def run_with(num_workers):
            data = _WaveformDataset(
                df, "label", _cfg(), rawboost=RawBoostParams(algo=3), rawboost_seed=123
            )
            kwargs = {"num_workers": num_workers}
            if num_workers:
                kwargs["persistent_workers"] = True
            loader = torch.utils.data.DataLoader(data, batch_size=1, **kwargs)
            return [wave[0].clone() for wave, _ in loader]

        single_process = run_with(0)
        multi_process = run_with(2)

        for a, b in zip(single_process, multi_process):
            assert torch.equal(a, b)

    def test_does_not_mutate_numpy_global_random_state(self, tmp_path):
        """Unlike the old np.random.seed()-based design, building/reading a
        seeded dataset must not touch numpy's global RNG -- that state is
        shared with every other test and unrelated code in the process."""
        df = _df(tmp_path)
        before = np.random.get_state()[1].copy()

        data = _WaveformDataset(
            df, "label", _cfg(), rawboost=RawBoostParams(algo=3), rawboost_seed=123
        )
        for i in range(len(df)):
            data[i]

        np.testing.assert_array_equal(before, np.random.get_state()[1])

    def test_seed_outside_numpy_range_does_not_crash(self):
        """MODEL.random_seed (parse_seed()) accepts any int, but
        np.random.RandomState only accepts a seed in [0, 2**32-1]; both the
        base seed and base_seed + idx must be reduced into that range."""
        _rawboost_rng(-5, 0)
        _rawboost_rng(2**32, 0)
        _rawboost_rng(2**32 - 1, idx=10)


class TestLoaders:
    def _model(self, tmp_path, rawboost, n_jobs=0, manual_seed=None):
        _util(tmp_path)
        model = AasistModel.__new__(AasistModel)
        model.util, model.n_jobs, model.target = Util("test"), n_jobs, "label"
        model.cfg = _cfg(rawboost)
        model._manual_seed = manual_seed
        return model

    def test_only_the_train_loader_adds_noise(self, tmp_path):
        model = self._model(tmp_path, RawBoostParams(algo=3))
        df = _df(tmp_path)
        train = model.get_loader(df, shuffle=True, train=True)
        test = model.get_loader(df, shuffle=False)
        assert train.dataset.rawboost is not None
        assert test.dataset.rawboost is None
        assert torch.equal(test.dataset[0][0], test.dataset[0][0])

    def test_off_means_no_noise_in_either_loader(self, tmp_path):
        model = self._model(tmp_path, None)
        df = _df(tmp_path)
        assert model.get_loader(df, shuffle=True, train=True).dataset.rawboost is None

    def test_rawboost_seed_set_for_seeded_train_loader(self, tmp_path):
        model = self._model(tmp_path, RawBoostParams(algo=3), manual_seed=123)
        df = _df(tmp_path)
        train = model.get_loader(df, shuffle=True, train=True)
        assert train.dataset.rawboost_seed == 123

    def test_no_rawboost_seed_without_a_seed(self, tmp_path):
        model = self._model(tmp_path, RawBoostParams(algo=3))
        df = _df(tmp_path)
        train = model.get_loader(df, shuffle=True, train=True)
        assert train.dataset.rawboost_seed is None

    def test_no_rawboost_seed_on_the_test_loader(self, tmp_path):
        model = self._model(tmp_path, RawBoostParams(algo=3), manual_seed=123)
        df = _df(tmp_path)
        test = model.get_loader(df, shuffle=False)
        assert test.dataset.rawboost_seed is None

    def test_rawboost_seed_set_regardless_of_n_jobs(self, tmp_path):
        """Unlike the old worker_init_fn, which only applied when
        n_jobs > 0, rawboost_seed is a dataset-level setting independent of
        the loader's process layout."""
        model = self._model(tmp_path, RawBoostParams(algo=3), n_jobs=0, manual_seed=123)
        df = _df(tmp_path)
        train = model.get_loader(df, shuffle=True, train=True)
        assert train.dataset.rawboost_seed == 123
