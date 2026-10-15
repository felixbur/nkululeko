"""Domain-balanced sampling (MODEL.domain_balanced_sampling) in the five
neural models, through the shared Model._make_train_loader().

mlp, mlp_reg and cnn are built with their real constructors; adm and aasist
(whose constructors need an SSL checkpoint / a full experiment) through their
get_loader(). Domains are "a" (6 rows) and "b" (2 rows); every batch of 4 must
hold two of each.
"""

import configparser
import os

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from PIL import Image

import nkululeko.glob_conf as glob_conf
from nkululeko.data.domain_sampler import DomainBalancedBatchSampler
from nkululeko.experiment_context import ExperimentContext
from nkululeko.models.aasist_config import AasistConfig
from nkululeko.models.domain_adversarial import DannConfig
from nkululeko.models.model_aasist import AasistModel
from nkululeko.models.model_adm import ADMModel
from nkululeko.models.model_cnn import CNNModel
from nkululeko.models.model_mlp import MLPModel
from nkululeko.models.model_mlp_regression import MLP_Reg_model
from nkululeko.utils.errors import NkululukoError
from nkululeko.utils.util import Util

N = 8
DOMAINS = ["a"] * 6 + ["b"] * 2
BATCH = 4


def _config(tmp_path, model_type, **model):
    config = configparser.ConfigParser()
    config["EXP"] = {
        "type": "regression" if model_type == "mlp_reg" else "classification",
        "name": "x",
        "root": str(tmp_path),
    }
    config["DATA"] = {"target": "label", "databases": "['a']"}
    config["FEATS"] = {"type": "['os']"}
    config["MODEL"] = {
        "type": model_type,
        "layers": "{'a': 8, 'b': 4}",
        "batch_size": str(BATCH),
        "n_jobs": "0",
        "device": "cpu",
        "domain_balanced_sampling": "True",
        **model,
    }
    glob_conf.config = config
    return config


@pytest.fixture(autouse=True)
def _reset_config():
    yield
    glob_conf.config = None


def _df(regression=False):
    labels = np.linspace(0, 1, N) if regression else [0, 1] * (N // 2)
    return pd.DataFrame({"label": labels, "source_db": DOMAINS})


def _images(tmp_path):
    files = []
    for i in range(N):
        path = tmp_path / f"{i}.png"
        Image.fromarray((np.random.rand(256, 256, 3) * 255).astype("uint8")).save(path)
        files.append(str(path))
    return pd.DataFrame({"f": files})


def _build(tmp_path, model_type, **model):
    config = _config(tmp_path, model_type, **model)
    ctx = ExperimentContext(
        config=config, labels=[0, 1], target="label", util=Util("test")
    )
    df = _df(regression=model_type == "mlp_reg")
    if model_type == "cnn":
        feats = _images(tmp_path)
        return CNNModel(df, df, feats, feats, context=ctx), df
    feats = pd.DataFrame(np.random.rand(N, 3), columns=["f1", "f2", "f3"])
    cls = MLP_Reg_model if model_type == "mlp_reg" else MLPModel
    return cls(df, df, feats, feats, context=ctx), df


def _assert_balanced(loader, df):
    assert isinstance(loader.batch_sampler, DomainBalancedBatchSampler)
    n_samples = 0
    for batch in loader.batch_sampler:
        assert len(batch) == BATCH
        assert pd.Series(
            df["source_db"].to_numpy()[batch]
        ).value_counts().to_dict() == {
            "a": 2,
            "b": 2,
        }
        n_samples += len(batch)
    assert len(loader) == len(loader.batch_sampler) == -(-N // BATCH)
    assert n_samples == len(loader) * BATCH


@pytest.mark.parametrize("model_type", ["mlp", "mlp_reg", "cnn"])
class TestRealConstructors:
    def test_trains_on_balanced_batches(self, tmp_path, model_type):
        model, df = _build(tmp_path, model_type)
        _assert_balanced(model.trainloader, df)
        before = [p.detach().clone() for p in model.model.parameters()]
        if model_type == "mlp_reg":
            model.train_epoch(model.model, model.trainloader, "cpu", model.optimizer)
        else:
            model.train()
        assert np.isfinite(model.loss)
        assert any(
            not torch.equal(p, b) for p, b in zip(model.model.parameters(), before)
        )

    def test_trains_with_dann_and_sam_too(self, tmp_path, model_type):
        model, df = _build(
            tmp_path, model_type, dann_columns="['source_db']", sam="True"
        )
        _assert_balanced(model.trainloader, df)
        head = model.dann_heads.head("source_db").classifier[0]
        before = head.weight.clone()
        if model_type == "mlp_reg":
            model.train_epoch(model.model, model.trainloader, "cpu", model.optimizer)
        else:
            model.train()
        assert np.isfinite(model.loss)
        assert not torch.allclose(head.weight, before)

    def test_predicting_the_train_data_sees_every_row_once_in_order(
        self, tmp_path, model_type
    ):
        model, df = _build(tmp_path, model_type)
        loader = model._eval_trainloader()
        assert loader is not model.trainloader
        assert loader.batch_size == BATCH and len(loader.dataset) == N
        seen = []

        class Stop(Exception):
            pass

        name = "evaluate_model" if model_type == "mlp_reg" else "evaluate"
        original = getattr(model, name)

        def spy(*args):
            seen.append(args[1])
            if args[1] is model.testloader:
                return original(*args)
            raise Stop

        setattr(model, name, spy)
        with pytest.raises(Stop):
            model.predict()
        assert seen == [model.testloader, loader]
        labels = torch.cat([batch[1] for batch in loader]).numpy()
        assert np.allclose(labels, df["label"].to_numpy())

    def test_off_by_default(self, tmp_path, model_type):
        model, _ = _build(tmp_path, model_type, domain_balanced_sampling="False")
        assert model.trainloader.batch_size == BATCH
        assert model._eval_trainloader() is model.trainloader

    def test_the_evaluation_of_the_train_data_matches_the_plain_loader(
        self, tmp_path, model_type
    ):
        if model_type == "mlp_reg":
            pytest.skip("mlp_reg: covered by the predict test")
        model, df = _build(tmp_path, model_type)
        uar, targets, predictions, logits = model.evaluate(
            model.model, model._eval_trainloader(), "cpu"
        )
        assert len(targets) == len(predictions) == N
        assert np.array_equal(targets.numpy(), df["label"].to_numpy())


class TestAdm:
    def _model(self, tmp_path, **model):
        _config(tmp_path, "adm", **model)
        m = ADMModel.__new__(ADMModel)
        m.util, m.batch_size, m.target = Util("test"), BATCH, "label"
        return m

    def test_get_loader_balances_only_the_train_loader(self, tmp_path):
        model, df = self._model(tmp_path), _df()
        feats = pd.DataFrame(np.random.rand(N, 5))
        train = model.get_loader(feats, df, True)
        _assert_balanced(train, df)
        plain = model.get_loader(feats, df, False)
        assert plain.batch_size == BATCH
        assert not isinstance(plain.batch_sampler, DomainBalancedBatchSampler)

    def test_batches_carry_the_domain_labels_of_their_rows(self, tmp_path):
        model, df = self._model(tmp_path), _df()
        feats = pd.DataFrame(np.arange(N * 3, dtype=float).reshape(N, 3))
        domains = (df["source_db"] == "b").astype(int).to_numpy().reshape(-1, 1)
        loader = model.get_loader(feats, df, True, domain_labels=domains)
        for features, labels, domain in loader:
            assert domain.reshape(-1).tolist() == [
                int(DOMAINS[int(f[0]) // 3] == "b") for f in features
            ]

    def test_predicting_uses_a_plain_loader(self, tmp_path):
        model, df = self._model(tmp_path), _df()
        feats = pd.DataFrame(np.random.rand(N, 5))
        model.trainloader = model.get_loader(feats, df, True)
        eval_loader = model._eval_trainloader()
        assert eval_loader.batch_size == BATCH and len(eval_loader.dataset) == N
        assert sum(len(b[1]) for b in eval_loader) == N


def _write_wavs(tmp_path):
    index = []
    for i in range(N):
        path = tmp_path / f"{i}.wav"
        sf.write(path, np.full(1600, 0.01 * (i + 1), dtype=np.float32), 16000)
        index.append((str(path), pd.Timedelta(0), pd.NaT))
    idx = pd.MultiIndex.from_tuples(index, names=["file", "start", "end"])
    return pd.DataFrame({"label": [0, 1] * (N // 2), "source_db": DOMAINS}, index=idx)


class TestAasist:
    def _model(self, tmp_path, n_jobs=0, **model):
        _config(tmp_path, "aasist", **model)
        m = AasistModel.__new__(AasistModel)
        m.util, m.n_jobs, m.target = Util("test"), n_jobs, "label"
        m.cfg = AasistConfig(
            device="cpu",
            ssl_model="x",
            max_len=1600,
            batch_size=BATCH,
            ssl_layer_pooling="last",
            freeze_ssl_frontend=False,
            dann=DannConfig(columns=[], lambda_=1.0, weight=1.0, reverse=True),
        )
        return m

    @pytest.mark.parametrize("n_jobs", [0, 2])
    def test_train_loader_yields_balanced_batches_of_real_audio(self, tmp_path, n_jobs):
        model, df = self._model(tmp_path, n_jobs), _write_wavs(tmp_path)
        loader = model.get_loader(df, shuffle=True, train=True)
        _assert_balanced(loader, df)
        for waveforms, labels in loader:
            assert waveforms.shape == (BATCH, 1600)
            # the mean of a waveform identifies its row: 0.01 * (row + 1)
            rows = np.round(waveforms.mean(dim=1).numpy() / 0.01).astype(int) - 1
            assert sorted(np.array(DOMAINS)[rows]) == ["a", "a", "b", "b"]

    def test_the_test_loader_is_plain(self, tmp_path):
        model, df = self._model(tmp_path), _write_wavs(tmp_path)
        loader = model.get_loader(df, shuffle=False)
        assert loader.batch_size == BATCH
        assert not isinstance(loader.batch_sampler, DomainBalancedBatchSampler)


class TestSharedHelper:
    def _model(self, tmp_path, **model):
        _config(tmp_path, "mlp", **model)
        m = MLPModel.__new__(MLPModel)
        m.util = Util("test")
        return m

    def test_a_missing_domain_column_is_an_error(self, tmp_path):
        model = self._model(tmp_path)
        df = _df().drop(columns="source_db")
        with pytest.raises(NkululukoError, match="source_db"):
            model._make_train_loader(list(range(N)), df, BATCH)

    def test_another_column_can_be_the_domain(self, tmp_path):
        model = self._model(tmp_path, domain_balanced_column="lang")
        df = _df().assign(lang=["de"] * 6 + ["en"] * 2)
        loader = model._make_train_loader(list(range(N)), df, BATCH)
        assert loader.batch_sampler.domains == ["de", "en"]

    def test_a_single_domain_falls_back_to_shuffling(self, tmp_path, caplog):
        model = self._model(tmp_path)
        df = _df().assign(source_db="a")
        loader = model._make_train_loader(list(range(N)), df, BATCH)
        assert loader.batch_size == BATCH
        assert not isinstance(loader.batch_sampler, DomainBalancedBatchSampler)
        assert "nothing to balance" in caplog.text

    def test_missing_values_are_an_error_even_with_one_domain_left(self, tmp_path):
        model = self._model(tmp_path)
        df = _df().assign(source_db="a")
        df.loc[0, "source_db"] = None  # one value and a gap: not "one domain"
        with pytest.raises(NkululukoError, match="no value"):
            model._make_train_loader(list(range(N)), df, BATCH)

    def test_rows_without_a_domain_are_an_error(self, tmp_path):
        model = self._model(tmp_path)
        df = _df()
        df.loc[0, "source_db"] = None
        with pytest.raises(NkululukoError, match="no value"):
            model._make_train_loader(list(range(N)), df, BATCH)

    def test_too_small_a_batch_is_an_error(self, tmp_path):
        model = self._model(tmp_path)
        with pytest.raises(NkululukoError, match="smaller than the number"):
            model._make_train_loader(list(range(N)), _df(), 1)

    def test_a_new_train_loader_forgets_the_old_eval_loader(self, tmp_path):
        model = self._model(tmp_path)
        model.trainloader = model._make_train_loader(list(range(N)), _df(), BATCH)
        first = model._eval_trainloader()
        model.trainloader = model._make_train_loader(list(range(N)), _df(), BATCH)
        assert model._eval_trainloader() is not first
