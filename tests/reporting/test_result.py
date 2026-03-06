"""Unit tests for Result class (nkululeko/reporting/result.py)."""

import configparser

import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.reporting.result import Result


@pytest.fixture(autouse=True)
def setup_glob_conf(tmp_path):
    config = configparser.ConfigParser()
    config["EXP"] = {"type": "classification", "name": "testexp", "root": str(tmp_path)}
    config["DATA"] = {"target": "emotion", "databases": "['emodb']"}
    config["MODEL"] = {"type": "xgb"}
    config["FEATS"] = {"type": "['os']"}
    glob_conf.config = config
    yield
    glob_conf.config = None


@pytest.fixture
def result():
    r = Result(test=0.75, train=0.80, loss=0.25, loss_eval=0.30, metric="UAR")
    r.set_upper_lower(upper=0.80, lower=0.70)
    return r


class TestResultInit:
    def test_attributes_set(self):
        r = Result(0.5, 0.6, 0.1, 0.2, "UAR")
        assert r.test == pytest.approx(0.5)
        assert r.train == pytest.approx(0.6)
        assert r.loss == pytest.approx(0.1)
        assert r.loss_eval == pytest.approx(0.2)
        assert r.metric == "UAR"

    def test_util_is_created(self):
        r = Result(0.5, 0.6, 0.1, 0.2, "UAR")
        assert r.util is not None


class TestGetResult:
    def test_returns_test_value(self, result):
        assert result.get_result() == result.test


class TestSetUpperLower:
    def test_upper_lower_stored(self, result):
        assert result.upper == pytest.approx(0.80)
        assert result.lower == pytest.approx(0.70)


class TestToString:
    def test_contains_all_fields(self, result):
        s = result.to_string()
        assert "test" in s
        assert "train" in s
        assert "loss" in s
        assert "UAR" in s


class TestGetTestResult:
    def test_format(self, result):
        s = result.get_test_result()
        assert "test" in s
        assert "UAR" in s
        assert "0.750" in s


class TestTestResultStr:
    def test_contains_metric_and_bounds(self, result):
        s = result.test_result_str()
        assert "UAR" in s
        assert "0.750" in s
        assert "0.800" in s
        assert "0.700" in s

    def test_format_parentheses(self, result):
        s = result.test_result_str()
        assert "(" in s and ")" in s
