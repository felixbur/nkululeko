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
    def debug(self, msg): pass
    def error(self, msg): raise Exception(msg)
    def get_path(self, key): return "./"
    def get_exp_name(self, only_train=False): return "exp"

@pytest.fixture(autouse=True)
def patch_globals(monkeypatch):
    # Patch global config and labels
    import nkululeko.glob_conf as glob_conf
    glob_conf.config = {
        "DATA": {"target": "label"},
        "MODEL": {"layers": "{'a': 8, 'b': 4}"}
    }
    glob_conf.labels = [0, 1]
    yield

@pytest.fixture
def dummy_data():
    # 4 samples, 3 features
    feats_train = pd.DataFrame(np.random.rand(4, 3), columns=['f1', 'f2', 'f3'])
    feats_test = pd.DataFrame(np.random.rand(2, 3), columns=['f1', 'f2', 'f3'])
    df_train = pd.DataFrame({'label': [0, 1, 0, 1]})
    df_test = pd.DataFrame({'label': [1, 0]})
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
        model.model = MLPModel.MLP(3, {'a': 8, 'b': 4}, 2, False).to("cpu")
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
    preds = mlp_model.get_predictions()
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == 2

def test_get_probas(mlp_model):
    mlp_model.train()
    _, _, _, logits = mlp_model.evaluate(mlp_model.model, mlp_model.testloader, mlp_model.device)
    probas = mlp_model.get_probas(logits)
    assert isinstance(probas, pd.DataFrame)
    assert set(probas.columns) == set([0, 1])

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