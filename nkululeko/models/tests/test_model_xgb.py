import pandas as pd
import pytest

from ..model_xgb import XGB_model


class DummyUtil:
    def config_val(self, section, key, default):
        return default
    def debug(self, msg):
        pass

class DummyModel(XGB_model):
    def __init__(self, df_train, df_test, feats_train, feats_test):
        # Patch util before calling super().__init__
        self.util = DummyUtil()
        self.target = "label"
        super().__init__(df_train, df_test, feats_train, feats_test)
        self.util = DummyUtil()
        self.target = "label"

@pytest.fixture
def dummy_data():
    df_train = pd.DataFrame({"label": [0, 1], "f1": [1.0, 2.0]})
    df_test = pd.DataFrame({"label": [0, 1], "f1": [1.5, 2.5]})
    feats_train = df_train[["f1"]]
    feats_test = df_test[["f1"]]
    return df_train, df_test, feats_train, feats_test

def test_get_type_returns_xgb(dummy_data):
    df_train, df_test, feats_train, feats_test = dummy_data
    model = DummyModel(df_train, df_test, feats_train, feats_test)
    assert model.get_type() == "xgb"