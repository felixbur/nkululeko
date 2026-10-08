"""Unit tests for AasistConfig (nkululeko/models/aasist_config.py)."""

import configparser

import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.models.aasist_config import AasistConfig
from nkululeko.utils.util import Util


def make_util(tmp_path, aasist_section=None, model_section=None):
    config = configparser.ConfigParser()
    config["EXP"] = {"type": "classification", "name": "testexp", "root": str(tmp_path)}
    config["DATA"] = {"target": "label", "databases": "['itw']"}
    config["MODEL"] = {"type": "aasist", **(model_section or {})}
    config["AASIST"] = aasist_section or {}
    config["FEATS"] = {"type": "[]"}
    glob_conf.config = config
    return Util("test")


@pytest.fixture(autouse=True)
def cleanup_glob_conf():
    yield
    glob_conf.config = None


class TestDefaults:
    def test_ssl_model_and_max_len_defaults(self, tmp_path):
        util = make_util(tmp_path)
        cfg = AasistConfig.from_util(util)
        assert cfg.ssl_model == "facebook/wav2vec2-xls-r-300m"
        assert cfg.max_len == 64600
        assert cfg.batch_size == 24

    def test_ssl_layer_pooling_and_freeze_defaults(self, tmp_path):
        util = make_util(tmp_path)
        cfg = AasistConfig.from_util(util)
        assert cfg.ssl_layer_pooling == "last"
        assert cfg.freeze_ssl_frontend is False


class TestOverrides:
    def test_ssl_model_and_max_len_overridable(self, tmp_path):
        util = make_util(
            tmp_path, {"ssl_model": "facebook/wav2vec2-base", "max_len": "32000"}
        )
        cfg = AasistConfig.from_util(util)
        assert cfg.ssl_model == "facebook/wav2vec2-base"
        assert cfg.max_len == 32000

    def test_ssl_layer_pooling_overridable(self, tmp_path):
        util = make_util(tmp_path, {"ssl_layer_pooling": "weighted"})
        cfg = AasistConfig.from_util(util)
        assert cfg.ssl_layer_pooling == "weighted"

    def test_freeze_ssl_frontend_overridable(self, tmp_path):
        util = make_util(tmp_path, {"freeze_ssl_frontend": "True"})
        cfg = AasistConfig.from_util(util)
        assert cfg.freeze_ssl_frontend is True

    def test_unknown_ssl_layer_pooling_raises(self, tmp_path):
        from nkululeko.utils.errors import NkululukoError

        util = make_util(tmp_path, {"ssl_layer_pooling": "bogus"})
        with pytest.raises(NkululukoError, match="ssl_layer_pooling"):
            AasistConfig.from_util(util)

    def test_device_override(self, tmp_path):
        # device reads from the shared [MODEL] section, not [AASIST] --
        # see AasistConfig.from_util's docstring for why.
        util = make_util(tmp_path)
        glob_conf.config["MODEL"]["device"] = "cpu"
        cfg = AasistConfig.from_util(util)
        assert cfg.device == "cpu"


class TestDannConfig:
    def test_defaults_are_off(self, tmp_path):
        cfg = AasistConfig.from_util(make_util(tmp_path))
        assert cfg.dann_columns == []
        assert cfg.dann_lambda == 1.0
        assert cfg.dann_weight == 1.0
        assert cfg.dann_reverse is True

    def test_reads_values_from_model_section(self, tmp_path):
        util = make_util(
            tmp_path,
            model_section={
                "dann_columns": "['source_db', 'language']",
                "dann_lambda": "0.5",
                "dann_weight": "2.0",
                "dann_reverse": "False",
            },
        )
        cfg = AasistConfig.from_util(util)
        assert cfg.dann_columns == ["source_db", "language"]
        assert cfg.dann_lambda == 0.5
        assert cfg.dann_weight == 2.0
        assert cfg.dann_reverse is False
