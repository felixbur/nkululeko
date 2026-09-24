import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch

from nkululeko.models.model_mlp import MLPModel


class DummyUtil:
    def config_val(self, section, key, default=None):
        # Provide defaults for required config values
        if key == "manual_seed":
            return True
        if key == "loss":
            return "cross"
        if key == "device":
            return "cpu"
        if key == "learning_rate":
            return 0.001
        if key == "batch_size":
            return 2
        if key == "drop":
            return False
        return default

    def debug(self, msg):
        pass

    def error(self, msg):
        raise Exception(msg)

    def get_path(self, key):
        return "./"

    def get_exp_name(self, only_train=False):
        return "exp"


@pytest.fixture(autouse=True)
def patch_globals(monkeypatch):
    # Patch global config and labels
    import nkululeko.glob_conf as glob_conf

    glob_conf.config = {
        "DATA": {"target": "label"},
        "MODEL": {"layers": "{'a': 8, 'b': 4}"},
    }
    glob_conf.labels = [0, 1]
    yield


@pytest.fixture
def dummy_data():
    # 4 samples, 3 features
    feats_train = pd.DataFrame(np.random.rand(4, 3), columns=["f1", "f2", "f3"])
    feats_test = pd.DataFrame(np.random.rand(2, 3), columns=["f1", "f2", "f3"])
    df_train = pd.DataFrame({"label": [0, 1, 0, 1]})
    df_test = pd.DataFrame({"label": [1, 0]})
    return df_train, df_test, feats_train, feats_test


@pytest.fixture
def mlp_model(dummy_data, monkeypatch):
    df_train, df_test, feats_train, feats_test = dummy_data
    with patch.object(MLPModel, "__init__", return_value=None):
        model = MLPModel(df_train, df_test, feats_train, feats_test)
        model.util = DummyUtil()
        model.n_jobs = 1
        model.target = "label"
        model.class_num = 2
        model.criterion = torch.nn.CrossEntropyLoss()
        model.device = "cpu"
        model.learning_rate = 0.001
        model.batch_size = 2
        model.num_workers = 1
        model.loss = 0.0
        model.loss_eval = 0.0
        model.run = 0
        model.epoch = 0
        model.df_test = df_test
        model.feats_test = feats_test
        model.feats_train = feats_train

        # Create a simple MLP model for testing
        model.model = MLPModel.MLP(3, {"a": 8, "b": 4}, 2, False, torch.nn.ReLU()).to(
            "cpu"
        )
        model.optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)

        # Create data loaders
        model.trainloader = model.get_loader(feats_train, df_train, True)
        model.testloader = model.get_loader(feats_test, df_test, False)
        model.store_path = "/tmp/test_model.pt"

        return model


def test_mlpmodel_init(mlp_model):
    assert hasattr(mlp_model, "model")
    assert hasattr(mlp_model, "trainloader")
    assert hasattr(mlp_model, "testloader")
    assert mlp_model.model is not None


def test_train_and_predict(mlp_model):
    mlp_model.train()
    report = mlp_model.predict()
    assert hasattr(report, "result")
    assert hasattr(report.result, "train")


def test_get_predictions(mlp_model):
    mlp_model.train()
    preds, probas = mlp_model.get_predictions()
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == 2


def test_get_probas(mlp_model):
    mlp_model.train()
    _, _, _, logits = mlp_model.evaluate(
        mlp_model.model, mlp_model.testloader, mlp_model.device
    )
    probas = mlp_model.get_probas(logits)
    assert isinstance(probas, pd.DataFrame)
    assert set(probas.columns) == set([0, 1])


class TestGetProbasAppliesSoftmax:
    """Regression (GH #446): get_probas() wrote raw logits into the
    per-class "probability" columns with no softmax. Logits aren't
    themselves probabilities (not bounded to [0, 1], don't sum to 1), so
    the saved columns - and anything derived from them (uncertainty,
    session averaging, calibration, a probability threshold on ROC) - were
    silently wrong. argmax (the predicted label) is unaffected either way,
    since softmax is monotonic.

    Row count matches mlp_model's df_test (2 rows, see dummy_data), since
    get_probas() indexes the result by self.df_test.index."""

    def test_probability_columns_are_valid_distributions(self, mlp_model):
        # Deliberately raw-logit-like values (negative, and > 1) that
        # would leak straight through into probas without softmax.
        logits = torch.tensor([[-2.0, 3.0], [1.5, -0.5]])

        probas = mlp_model.get_probas(logits)

        row_sums = probas.sum(axis=1).to_numpy()
        np.testing.assert_allclose(row_sums, np.ones(2), rtol=1e-5)
        assert (probas.to_numpy() >= 0).all()
        assert (probas.to_numpy() <= 1).all()

    def test_matches_manual_softmax(self, mlp_model):
        logits = torch.tensor([[-2.0, 3.0], [1.5, -0.5]])
        expected = torch.softmax(logits, dim=1).numpy()

        probas = mlp_model.get_probas(logits)

        np.testing.assert_allclose(probas[0].to_numpy(), expected[:, 0], rtol=1e-5)
        np.testing.assert_allclose(probas[1].to_numpy(), expected[:, 1], rtol=1e-5)

    def test_argmax_label_unaffected(self, mlp_model):
        logits = torch.tensor([[-2.0, 3.0], [1.5, -0.5]])

        probas = mlp_model.get_probas(logits)

        expected_argmax = logits.argmax(dim=1).numpy()
        actual_argmax = probas.to_numpy().argmax(axis=1)
        np.testing.assert_array_equal(actual_argmax, expected_argmax)


def test_predict_sample(mlp_model):
    mlp_model.train()
    feats = np.random.rand(3)
    res = mlp_model.predict_sample(feats)
    assert isinstance(res, dict)
    assert set(res.keys()) == set([0, 1])


def test_predict_shap(mlp_model):
    mlp_model.train()
    feats = pd.DataFrame(np.random.rand(2, 3))
    results = mlp_model.predict_shap(feats)
    assert len(results) == 2


def test_store_and_load(tmp_path, mlp_model, monkeypatch):
    mlp_model.train()

    # Mock the util methods that load() uses to construct the path
    def mock_get_path(key):
        if key == "model_dir":
            return str(tmp_path) + "/"
        return "./"

    def mock_get_exp_name(only_train=False):
        return "model"

    mlp_model.util.get_path = mock_get_path
    mlp_model.util.get_exp_name = mock_get_exp_name

    # Set store path to match what load() will construct
    mlp_model.store_path = str(tmp_path) + "/model_0_000.model"
    mlp_model.store()

    # Simulate loading
    mlp_model.load(0, 0)
    assert mlp_model.model is not None


def test_set_testdata(mlp_model, dummy_data):
    _, df_test, _, feats_test = dummy_data
    mlp_model.set_testdata(df_test, feats_test)
    assert mlp_model.testloader is not None


def test_reset_test(mlp_model, dummy_data):
    _, df_test, _, feats_test = dummy_data
    mlp_model.reset_test(df_test, feats_test)
    assert mlp_model.testloader is not None


def test_mlp_model_init_with_dropout_list():
    # Test with a list of dropout values
    mlp_inner = MLPModel.MLP(3, {"a": 8, "b": 4}, 2, [0.1, 0.2], torch.nn.ReLU())
    dropout_layers = [l for l in mlp_inner.linear if isinstance(l, torch.nn.Dropout)]
    assert len(dropout_layers) == 1
    assert dropout_layers[0].p == pytest.approx(0.1)


def test_mlp_model_init_with_dropout_float():
    # Test with a single float value for dropout
    mlp_inner = MLPModel.MLP(3, {"a": 8, "b": 4}, 2, 0.5, torch.nn.ReLU())
    dropout_layers = [l for l in mlp_inner.linear if isinstance(l, torch.nn.Dropout)]
    assert len(dropout_layers) == 1
    assert dropout_layers[0].p == pytest.approx(0.5)


class TestGetLoaderIsLinearNotQuadratic:
    """Regression (GH #444): get_loader() called df_x.values once *per
    row* inside its Python loop. df_x.values re-materializes the whole
    (n_samples, n_features) array on every access; when the DataFrame
    isn't a single consolidated block (e.g. after feature balancing
    concatenates several), that materialization is itself O(n), making
    the whole loop O(n^2) -- 16114 rows after `balancing = ros` took over
    10 minutes and 22 GB RSS; ~30s once hoisted outside the loop.

    Counts real .values property accesses directly (not wall-clock time),
    since reliably forcing pandas' internal block fragmentation from a
    unit test is brittle/pandas-version-dependent, while the actual
    defect - .values evaluated N times instead of once - is not."""

    def test_values_accessed_a_bounded_number_of_times_not_once_per_row(
        self, mlp_model, dummy_data
    ):
        df_train, _df_test, feats_train, _feats_test = dummy_data
        real_values = pd.DataFrame.values
        access_count = 0

        def counting_values(self):
            nonlocal access_count
            access_count += 1
            return real_values.fget(self)

        with patch.object(pd.DataFrame, "values", property(counting_values)):
            mlp_model.get_loader(feats_train, df_train, shuffle=False)

        # Exactly one access for df_x.values; a couple more (not N) are
        # fine for df_y[self.target] internals - the point is this must
        # not scale with len(df_x) (4 rows here; the bug would have made
        # this at least 4, growing linearly with row count on real data).
        assert access_count < len(feats_train)

    def test_loader_still_yields_correct_rows_in_order(self, mlp_model, dummy_data):
        df_train, _df_test, feats_train, _feats_test = dummy_data
        loader = mlp_model.get_loader(feats_train, df_train, shuffle=False)

        seen_features = []
        seen_labels = []
        for batch_x, batch_y in loader:
            seen_features.append(batch_x.numpy())
            seen_labels.append(batch_y.numpy())
        features = np.concatenate(seen_features)
        labels = np.concatenate(seen_labels)

        np.testing.assert_allclose(features, feats_train.values, rtol=1e-5)
        np.testing.assert_array_equal(labels, df_train["label"].values)
