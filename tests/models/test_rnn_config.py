"""Unit tests for RNNConfig (nkululeko/models/rnn_config.py)."""

import configparser

import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.models.rnn_config import RNNConfig
from nkululeko.utils.util import Util


def make_util(tmp_path, rnn_section=None, model_section=None):
    config = configparser.ConfigParser()
    config["EXP"] = {"type": "classification", "name": "testexp", "root": str(tmp_path)}
    config["DATA"] = {"target": "label", "databases": "['itw']"}
    config["MODEL"] = {"type": "lstm", **(model_section or {})}
    config["RNN"] = rnn_section or {}
    config["FEATS"] = {"type": "[]"}
    glob_conf.config = config
    return Util("test")


@pytest.fixture(autouse=True)
def cleanup_glob_conf():
    yield
    glob_conf.config = None


class TestDefaults:
    def test_feature_source_and_mel_defaults(self, tmp_path):
        util = make_util(tmp_path)
        cfg = RNNConfig.from_util(util)
        assert cfg.feature_source == "melspec"
        assert cfg.n_mels == 64
        assert cfg.win_samples == 400  # 25ms @ 16kHz
        assert cfg.hop_samples == 160  # 10ms @ 16kHz

    def test_opensmile_set_default(self, tmp_path):
        util = make_util(tmp_path)
        cfg = RNNConfig.from_util(util)
        assert cfg.opensmile_set == "eGeMAPSv02"

    def test_architecture_defaults(self, tmp_path):
        util = make_util(tmp_path)
        cfg = RNNConfig.from_util(util)
        assert cfg.max_len == 64600
        assert cfg.hidden_size == 64
        assert cfg.num_layers == 2
        assert cfg.bidirectional is False
        assert cfg.batch_size == 32
        assert cfg.dropout == 0.0


class TestOverrides:
    def test_feature_source_overridable(self, tmp_path):
        util = make_util(tmp_path, {"feature_source": "opensmile_lld"})
        cfg = RNNConfig.from_util(util)
        assert cfg.feature_source == "opensmile_lld"

    def test_unknown_feature_source_raises(self, tmp_path):
        from nkululeko.utils.errors import NkululukoError

        util = make_util(tmp_path, {"feature_source": "bogus"})
        with pytest.raises(NkululukoError, match="feature_source"):
            RNNConfig.from_util(util)

    def test_mel_params_overridable(self, tmp_path):
        util = make_util(tmp_path, {"n_mels": "80", "win_dur": "50", "hop_dur": "20"})
        cfg = RNNConfig.from_util(util)
        assert cfg.n_mels == 80
        assert cfg.win_samples == 800
        assert cfg.hop_samples == 320

    def test_unknown_opensmile_set_warns_not_raises(self, tmp_path):
        util = make_util(tmp_path, {"opensmile_set": "bogus"})
        cfg = RNNConfig.from_util(util)  # should not raise
        assert cfg.opensmile_set == "bogus"

    def test_architecture_overridable(self, tmp_path):
        util = make_util(
            tmp_path,
            {
                "max_len": "32000",
                "hidden_size": "128",
                "num_layers": "3",
                "bidirectional": "True",
                "batch_size": "16",
            },
        )
        cfg = RNNConfig.from_util(util)
        assert cfg.max_len == 32000
        assert cfg.hidden_size == 128
        assert cfg.num_layers == 3
        assert cfg.bidirectional is True
        assert cfg.batch_size == 16

    def test_dropout_reads_shared_model_drop(self, tmp_path):
        util = make_util(tmp_path, model_section={"drop": "0.3"})
        cfg = RNNConfig.from_util(util)
        assert cfg.dropout == 0.3

    def test_device_override(self, tmp_path):
        # device reads from the shared [MODEL] section, not [RNN].
        util = make_util(tmp_path)
        glob_conf.config["MODEL"]["device"] = "cpu"
        cfg = RNNConfig.from_util(util)
        assert cfg.device == "cpu"


class TestDannConfig:
    def test_defaults_are_off(self, tmp_path):
        dann = RNNConfig.from_util(make_util(tmp_path)).dann
        assert dann.columns == []
        assert dann.lambda_ == 1.0
        assert dann.weight == 1.0
        assert dann.reverse is True

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
        dann = RNNConfig.from_util(util).dann
        assert dann.columns == ["source_db", "language"]
        assert dann.lambda_ == 0.5
        assert dann.weight == 2.0
        assert dann.reverse is False

    def test_unquoted_columns_is_a_clean_error(self, tmp_path):
        from nkululeko.utils.errors import NkululukoError

        util = make_util(tmp_path, model_section={"dann_columns": "source_db"})
        with pytest.raises(NkululukoError, match="dann_columns"):
            RNNConfig.from_util(util)
