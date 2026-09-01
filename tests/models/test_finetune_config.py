"""Unit tests for FinetuneConfig (nkululeko/models/finetune_config.py)."""

import configparser

import pytest
import torch

import nkululeko.glob_conf as glob_conf
from nkululeko.models.finetune_config import FinetuneConfig
from nkululeko.utils.util import Util


def make_util(tmp_path, finetune_section=None):
    config = configparser.ConfigParser()
    config["EXP"] = {"type": "classification", "name": "testexp", "root": str(tmp_path)}
    config["DATA"] = {"target": "emotion", "databases": "['emodb']"}
    config["MODEL"] = {"type": "finetune"}
    config["FINETUNE"] = finetune_section or {}
    config["FEATS"] = {"type": "['os']"}
    glob_conf.config = config
    return Util("test")


@pytest.fixture(autouse=True)
def cleanup_glob_conf():
    yield
    glob_conf.config = None


class TestDefaults:
    def test_classifier_loss_and_measure_defaults(self, tmp_path):
        util = make_util(tmp_path)
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.loss == "cross"
        assert cfg.measure == "uar"

    def test_regressor_loss_and_measure_defaults(self, tmp_path):
        util = make_util(tmp_path)
        cfg = FinetuneConfig.from_util(util, is_classifier=False)
        assert cfg.loss == "1-ccc"
        assert cfg.measure == "ccc"

    def test_measure_not_configurable_for_classifier(self, tmp_path):
        util = make_util(tmp_path, {"measure": "mse"})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.measure == "uar"

    def test_batch_size_and_learning_rate_defaults(self, tmp_path):
        util = make_util(tmp_path)
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.batch_size == 8
        assert cfg.learning_rate == 0.0001
        assert cfg.max_duration == 8.0
        assert cfg.pretrained_model == "facebook/wav2vec2-large-robust-ft-swbd-300h"


class TestDrop:
    def test_drop_default_is_zero(self, tmp_path):
        util = make_util(tmp_path)
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.drop == 0.0

    def test_drop_explicit_value(self, tmp_path):
        util = make_util(tmp_path, {"drop": "0.3"})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.drop == 0.3

    def test_drop_empty_string_uses_default(self, tmp_path):
        util = make_util(tmp_path, {"drop": ""})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.drop == 0.0

    def test_drop_whitespace_only_uses_default(self, tmp_path):
        util = make_util(tmp_path, {"drop": "   "})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.drop == 0.0


class TestDevice:
    def test_device_autodetect(self, tmp_path):
        util = make_util(tmp_path)
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert cfg.device == expected

    def test_device_explicit_passthrough(self, tmp_path):
        util = make_util(tmp_path, {"device": "4"})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.device == "4"


class TestFreezeLayers:
    def test_freeze_layers_default_zero(self, tmp_path):
        util = make_util(tmp_path)
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.freeze_layers == 0

    def test_freeze_layers_explicit_value(self, tmp_path):
        util = make_util(tmp_path, {"freeze_layers": "6"})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.freeze_layers == 6

    def test_freeze_layers_empty_string_uses_default(self, tmp_path):
        util = make_util(tmp_path, {"freeze_layers": ""})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.freeze_layers == 0

    def test_freeze_layers_whitespace_only_uses_default(self, tmp_path):
        util = make_util(tmp_path, {"freeze_layers": "   "})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.freeze_layers == 0


class TestNumLayers:
    def test_num_layers_default_none(self, tmp_path):
        util = make_util(tmp_path)
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.num_layers is None

    def test_num_layers_explicit_value(self, tmp_path):
        util = make_util(tmp_path, {"num_layers": "6"})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.num_layers == 6

    def test_num_layers_empty_string_stays_none(self, tmp_path):
        util = make_util(tmp_path, {"num_layers": ""})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.num_layers is None

    def test_num_layers_whitespace_only_stays_none(self, tmp_path):
        util = make_util(tmp_path, {"num_layers": "   "})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.num_layers is None


class TestHeadLayers:
    def test_head_layers_default_none(self, tmp_path):
        util = make_util(tmp_path)
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.head_layers is None

    def test_head_layers_explicit_value(self, tmp_path):
        util = make_util(tmp_path, {"head_layers": "[1024, 256]"})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.head_layers == [1024, 256]

    def test_head_layers_empty_string_stays_none(self, tmp_path):
        util = make_util(tmp_path, {"head_layers": ""})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.head_layers is None


class TestHeadActivation:
    def test_head_activation_default_tanh(self, tmp_path):
        util = make_util(tmp_path)
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.head_activation == "tanh"

    def test_head_activation_explicit_value(self, tmp_path):
        util = make_util(tmp_path, {"head_activation": "relu"})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.head_activation == "relu"


class TestPooling:
    def test_pooling_default_mean(self, tmp_path):
        util = make_util(tmp_path)
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.pooling == "mean"

    def test_pooling_explicit_meanvar(self, tmp_path):
        util = make_util(tmp_path, {"pooling": "meanvar"})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.pooling == "meanvar"

    def test_pooling_unknown_value_raises(self, tmp_path):
        from nkululeko.utils.util import NkululukoError

        util = make_util(tmp_path, {"pooling": "max"})
        with pytest.raises(NkululukoError):
            FinetuneConfig.from_util(util, is_classifier=True)


class TestWarmupRatio:
    def test_warmup_ratio_default_zero(self, tmp_path):
        util = make_util(tmp_path)
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.warmup_ratio == 0.0

    def test_warmup_ratio_explicit_value(self, tmp_path):
        util = make_util(tmp_path, {"warmup_ratio": "0.1"})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.warmup_ratio == 0.1


class TestLayerPooling:
    def test_layer_pooling_default_last(self, tmp_path):
        util = make_util(tmp_path)
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.layer_pooling == "last"

    def test_layer_pooling_explicit_weighted(self, tmp_path):
        util = make_util(tmp_path, {"layer_pooling": "weighted"})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.layer_pooling == "weighted"

    def test_layer_pooling_unknown_value_raises(self, tmp_path):
        from nkululeko.utils.util import NkululukoError

        util = make_util(tmp_path, {"layer_pooling": "first"})
        with pytest.raises(NkululukoError):
            FinetuneConfig.from_util(util, is_classifier=True)


class TestClassWeight:
    def test_class_weight_default_false(self, tmp_path):
        util = make_util(tmp_path)
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.class_weight is False

    def test_class_weight_true(self, tmp_path):
        util = make_util(tmp_path, {"class_weight": "True"})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.class_weight is True

    def test_class_weight_false_string_stays_false(self, tmp_path):
        util = make_util(tmp_path, {"class_weight": "False"})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.class_weight is False


class TestBalancing:
    def test_balancing_default_false(self, tmp_path):
        util = make_util(tmp_path)
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.balancing is False

    def test_balancing_algorithm_name_passes_through(self, tmp_path):
        util = make_util(tmp_path, {"balancing": "smote"})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.balancing == "smote"

    def test_balancing_false_string_becomes_false(self, tmp_path):
        util = make_util(tmp_path, {"balancing": "False"})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.balancing is False

    def test_balancing_none_string_becomes_false(self, tmp_path):
        util = make_util(tmp_path, {"balancing": "none"})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.balancing is False

    def test_balancing_empty_string_becomes_false(self, tmp_path):
        util = make_util(tmp_path, {"balancing": ""})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.balancing is False


class TestPushToHub:
    def test_push_to_hub_default_false(self, tmp_path):
        util = make_util(tmp_path)
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.push_to_hub is False

    def test_push_to_hub_true(self, tmp_path):
        util = make_util(tmp_path, {"push_to_hub": "True"})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.push_to_hub is True

    def test_push_to_hub_false_string(self, tmp_path):
        util = make_util(tmp_path, {"push_to_hub": "False"})
        cfg = FinetuneConfig.from_util(util, is_classifier=True)
        assert cfg.push_to_hub is False
