"""Unit test for CNNModel.load_path() (nkululeko/models/model_cnn.py).

GH #477: load_path() called self.MLP(...), which doesn't exist on
CNNModel - every call raised AttributeError before the state_dict was
even loaded. Its sibling load(), two methods above, already builds the
model correctly (myCNN(layers, self.class_num)); load_path() now does
the same.
"""

import pytest
import torch

from nkululeko.models.model_cnn import CNNModel, myCNN


class _Util:
    def config_val(self, section, key, default=None):
        return default

    def debug(self, msg):
        pass

    def get_path(self, key):
        return "./"

    def get_exp_name(self, only_train=False):
        return "exp"


@pytest.fixture
def stored_model_path(tmp_path):
    net = myCNN({"l1": 120, "l2": 84}, 3)
    path = tmp_path / "model.pt"
    torch.save(net.state_dict(), str(path))
    return str(path)


def test_load_path_builds_a_usable_model(stored_model_path):
    model = CNNModel.__new__(CNNModel)
    model.util = _Util()
    model.context = type("Ctx", (), {"config": {"MODEL": {"layers": "{'l1': 120, 'l2': 84}"}}})()
    model.class_num = 3

    model.load_path(stored_model_path, run=0, epoch=0)

    assert isinstance(model.model, myCNN)
    images = torch.rand(2, 3, 256, 256)
    logits = model.model(images)
    assert logits.shape == (2, 3)


def test_load_path_missing_file_raises_file_not_found_error():
    model = CNNModel.__new__(CNNModel)
    model.util = _Util()

    with pytest.raises(FileNotFoundError):
        model.load_path("/nonexistent/path/model.pt", run=0, epoch=0)
