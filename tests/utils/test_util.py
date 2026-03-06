"""Unit tests for Util class (nkululeko/utils/util.py)."""

import configparser

import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.utils.util import Util


@pytest.fixture(autouse=True)
def setup_glob_conf():
    """Provide a minimal valid config for every test."""
    config = configparser.ConfigParser()
    config["EXP"] = {
        "type": "classification",
        "name": "testexp",
        "root": "/tmp",
    }
    config["DATA"] = {
        "target": "emotion",
        "databases": "['emodb']",
    }
    config["MODEL"] = {"type": "xgb"}
    config["FEATS"] = {"type": "['os']"}
    glob_conf.config = config
    yield
    glob_conf.config = None


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestUtilInit:
    def test_caller_stored(self):
        u = Util("mycaller")
        assert u.caller == "mycaller"

    def test_config_loaded_from_glob_conf(self):
        u = Util("test")
        assert u.config is not None

    def test_no_config_mode(self):
        u = Util("test", has_config=False)
        assert u.config is None

    def test_logger_created(self):
        u = Util("test")
        assert u.logger is not None


# ---------------------------------------------------------------------------
# config_val / config_val_list
# ---------------------------------------------------------------------------


class TestConfigVal:
    def test_returns_existing_value(self):
        u = Util("test")
        assert u.config_val("EXP", "type", "regression") == "classification"

    def test_returns_default_for_missing_key(self):
        u = Util("test")
        assert u.config_val("EXP", "no_such_key", "fallback") == "fallback"

    def test_returns_default_when_no_config(self):
        u = Util("test", has_config=False)
        assert u.config_val("EXP", "type", "fallback") == "fallback"

    def test_config_val_list_parses_list(self):
        u = Util("test")
        result = u.config_val_list("DATA", "databases", [])
        assert result == ["emodb"]

    def test_config_val_list_returns_default_for_missing(self):
        u = Util("test")
        result = u.config_val_list("DATA", "missing_key", ["default_item"])
        assert result == ["default_item"]


# ---------------------------------------------------------------------------
# set_config_val
# ---------------------------------------------------------------------------


class TestSetConfigVal:
    def test_set_existing_section_key(self):
        u = Util("test")
        u.set_config_val("EXP", "type", "regression")
        assert u.config["EXP"]["type"] == "regression"

    def test_set_new_section_creates_it(self):
        u = Util("test")
        u.set_config_val("NEWSEC", "mykey", "myval")
        assert u.config["NEWSEC"]["mykey"] == "myval"


# ---------------------------------------------------------------------------
# exp_is_classification
# ---------------------------------------------------------------------------


class TestExpIsClassification:
    def test_classification_returns_true(self):
        u = Util("test")
        assert u.exp_is_classification() is True

    def test_regression_returns_false(self):
        glob_conf.config["EXP"]["type"] = "regression"
        u = Util("test")
        assert u.exp_is_classification() is False


# ---------------------------------------------------------------------------
# high_is_good
# ---------------------------------------------------------------------------


class TestHighIsGood:
    def test_classification_uar_is_high_good(self):
        u = Util("test")
        assert u.high_is_good() is True

    def test_classification_eer_is_not_high_good(self):
        glob_conf.config["MODEL"]["measure"] = "eer"
        u = Util("test")
        assert u.high_is_good() is False

    def test_regression_mse_is_not_high_good(self):
        glob_conf.config["EXP"]["type"] = "regression"
        glob_conf.config["MODEL"]["measure"] = "mse"
        u = Util("test")
        assert u.high_is_good() is False

    def test_regression_mae_is_not_high_good(self):
        glob_conf.config["EXP"]["type"] = "regression"
        glob_conf.config["MODEL"]["measure"] = "mae"
        u = Util("test")
        assert u.high_is_good() is False

    def test_regression_ccc_is_high_good(self):
        glob_conf.config["EXP"]["type"] = "regression"
        glob_conf.config["MODEL"]["measure"] = "ccc"
        u = Util("test")
        assert u.high_is_good() is True


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


class TestNumericHelpers:
    def test_to_3_digits_truncates(self):
        u = Util("test")
        assert u.to_3_digits(0.12345) == 0.123

    def test_to_3_digits_str_format(self):
        u = Util("test")
        assert u.to_3_digits_str(0.12345) == "0.123"

    def test_to_3_digits_str_zero(self):
        u = Util("test")
        assert u.to_3_digits_str(0.0) == "0.000"

    def test_to_4_digits_truncates(self):
        u = Util("test")
        assert u.to_4_digits(0.123456) == 0.1234

    def test_to_4_digits_str_format(self):
        u = Util("test")
        assert u.to_4_digits_str(0.12345) == "0.1234"

    def test_to_4_digits_str_nan(self):
        import math

        u = Util("test")
        assert u.to_4_digits_str(float("nan")) == "nan"
