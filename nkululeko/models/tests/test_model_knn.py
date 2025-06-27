from unittest.mock import MagicMock, patch

import pytest

from nkululeko.models.model_knn import KNN_model


@pytest.fixture
def mock_util():
    mock = MagicMock()
    mock.config_val.side_effect = lambda section, key, default: {
        ("MODEL", "KNN_weights", "uniform"): "distance",
        ("MODEL", "K_val", "5"): "3"
    }[(section, key, default)]
    return mock

@pytest.fixture
def dummy_data():
    df_train = MagicMock()
    df_test = MagicMock()
    feats_train = MagicMock()
    feats_test = MagicMock()
    return df_train, df_test, feats_train, feats_test

def test_knn_model_initialization(monkeypatch, mock_util, dummy_data):
    with patch.object(KNN_model, "__init__", return_value=None):
        model = KNN_model(*dummy_data)
        model.util = mock_util
        model.name = "knn"
        from sklearn.neighbors import KNeighborsClassifier
        model.clf = KNeighborsClassifier(n_neighbors=3, weights="distance")
        model.is_classifier = True
        assert model.name == "knn"
        assert model.clf.get_params()["n_neighbors"] == 3
        assert model.clf.get_params()["weights"] == "distance"
        assert model.is_classifier is True

def test_knn_model_default_params(monkeypatch, dummy_data):
    mock_util = MagicMock()
    mock_util.config_val.side_effect = lambda section, key, default: default
    with patch.object(KNN_model, "__init__", return_value=None):
        model = KNN_model(*dummy_data)
        model.util = mock_util
        model.name = "knn"
        from sklearn.neighbors import KNeighborsClassifier
        model.clf = KNeighborsClassifier(n_neighbors=5, weights="uniform")
        model.is_classifier = True
        assert model.clf.get_params()["n_neighbors"] == 5
        assert model.clf.get_params()["weights"] == "uniform"