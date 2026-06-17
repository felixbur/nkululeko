"""Unit tests for Util class (nkululeko/utils/util.py)."""

import configparser
import logging

import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.utils.util import Util


@pytest.fixture(autouse=True)
def setup_glob_conf(tmp_path):
    """Provide a minimal valid config for every test."""
    config = configparser.ConfigParser()
    config["EXP"] = {
        "type": "classification",
        "name": "testexp",
        "root": str(tmp_path),
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
# config_val_bool
# ---------------------------------------------------------------------------


class TestConfigValBool:
    @pytest.mark.parametrize("value", ["True", "TRUE", "1", "yes", "  yes  "])
    def test_truthy_string_values(self, value):
        glob_conf.config["FEATS"]["no_reuse"] = value
        u = Util("test")
        assert u.config_val_bool("FEATS", "no_reuse", False) is True

    @pytest.mark.parametrize("value", ["False", "  False  "])
    def test_falsy_string_values(self, value):
        glob_conf.config["FEATS"]["no_reuse"] = value
        u = Util("test")
        assert u.config_val_bool("FEATS", "no_reuse", True) is False

    def test_returns_default_for_missing_key(self):
        u = Util("test")
        assert u.config_val_bool("FEATS", "nonexistent", False) is False
        assert u.config_val_bool("FEATS", "nonexistent", True) is True

    def test_returns_existing_bool_value(self):
        # ConfigParser stores strings only; verify bool defaults convert correctly
        u = Util("test", has_config=False)
        assert u.config_val_bool("FEATS", "no_reuse", True) is True
        assert u.config_val_bool("FEATS", "no_reuse", False) is False

    def test_rejects_arbitrary_code(self):
        glob_conf.config["FEATS"]["no_reuse"] = "__import__('os').system('echo hacked')"
        u = Util("test")
        # Should safely return False, not execute code
        assert u.config_val_bool("FEATS", "no_reuse", False) is False


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
        assert u.to_3_digits(0.12345) == pytest.approx(0.123)

    def test_to_3_digits_str_format(self):
        u = Util("test")
        assert u.to_3_digits_str(0.12345) == "0.123"

    def test_to_3_digits_str_zero(self):
        u = Util("test")
        assert u.to_3_digits_str(0.0) == "0.000"

    def test_to_4_digits_truncates(self):
        u = Util("test")
        assert u.to_4_digits(0.123456) == pytest.approx(0.1234)

    def test_to_4_digits_str_format(self):
        u = Util("test")
        assert u.to_4_digits_str(0.12345) == "0.1234"

    def test_to_4_digits_str_nan(self):

        u = Util("test")
        assert u.to_4_digits_str(float("nan")) == "nan"


# ---------------------------------------------------------------------------
# setup_logging / file handler
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def _reset_logger(self):
        """Remove all handlers from the shared module logger between tests."""
        import nkululeko.utils.util as util_mod

        logger = logging.getLogger(util_mod.__name__)
        for h in logger.handlers[:]:
            h.close()
            logger.removeHandler(h)

    def setup_method(self):
        self._reset_logger()

    def teardown_method(self):
        self._reset_logger()

    def test_file_handler_created_when_exp_config_present(self, tmp_path):
        import nkululeko.utils.util as util_mod

        glob_conf.config["EXP"]["root"] = str(tmp_path)
        glob_conf.config["EXP"]["name"] = "logtest"
        Util("test")
        logger = logging.getLogger(util_mod.__name__)
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename.endswith(".log")

    def test_no_duplicate_file_handler_on_second_util(self, tmp_path):
        import nkululeko.utils.util as util_mod

        glob_conf.config["EXP"]["root"] = str(tmp_path)
        glob_conf.config["EXP"]["name"] = "logtest"
        Util("test")
        Util("test2")
        logger = logging.getLogger(util_mod.__name__)
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1

    def test_file_handler_replaced_when_experiment_changes(self, tmp_path):
        import nkululeko.utils.util as util_mod

        # First experiment
        glob_conf.config["EXP"]["root"] = str(tmp_path)
        glob_conf.config["EXP"]["name"] = "exp_one"
        Util("test")
        logger = logging.getLogger(util_mod.__name__)
        first_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(first_handlers) == 1
        first_path = first_handlers[0].baseFilename

        # Second experiment with different name
        glob_conf.config["EXP"]["name"] = "exp_two"
        Util("test")
        second_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(second_handlers) == 1
        assert second_handlers[0].baseFilename != first_path
        assert "exp_two" in second_handlers[0].baseFilename

    def test_no_file_handler_without_config(self, tmp_path):
        import nkululeko.utils.util as util_mod

        glob_conf.config = None
        Util("test", has_config=False)
        logger = logging.getLogger(util_mod.__name__)
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 0

    def test_oserror_falls_back_to_console_only(self, tmp_path, monkeypatch):
        import nkululeko.utils.util as util_mod
        import audeer

        glob_conf.config["EXP"]["root"] = str(tmp_path)
        glob_conf.config["EXP"]["name"] = "logtest"

        def raise_oserror(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(audeer, "mkdir", raise_oserror)
        Util("test")  # Should not raise
        logger = logging.getLogger(util_mod.__name__)
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 0


# ---------------------------------------------------------------------------
# get_path()
# ---------------------------------------------------------------------------


class TestGetPath:
    def test_no_config_fig_dir_default(self, tmp_path, monkeypatch):
        glob_conf.config = None
        monkeypatch.chdir(tmp_path)
        u = Util("test", has_config=False)
        path = u.get_path("fig_dir")
        assert path == "./images/"
        assert (tmp_path / "images").is_dir()

    def test_no_config_res_dir_default(self, tmp_path, monkeypatch):
        glob_conf.config = None
        monkeypatch.chdir(tmp_path)
        u = Util("test", has_config=False)
        path = u.get_path("res_dir")
        assert path == "./results/"
        assert (tmp_path / "results").is_dir()

    def test_no_config_model_dir_default(self, tmp_path, monkeypatch):
        glob_conf.config = None
        monkeypatch.chdir(tmp_path)
        u = Util("test", has_config=False)
        path = u.get_path("model_dir")
        assert path == "./models/"
        assert (tmp_path / "models").is_dir()

    def test_no_config_cache_default(self, tmp_path, monkeypatch):
        glob_conf.config = None
        monkeypatch.chdir(tmp_path)
        u = Util("test", has_config=False)
        path = u.get_path("cache")
        assert path == "./cache/"
        assert (tmp_path / "cache").is_dir()

    def test_no_config_unknown_entry_returns_store(self, tmp_path, monkeypatch):
        glob_conf.config = None
        monkeypatch.chdir(tmp_path)
        u = Util("test", has_config=False)
        path = u.get_path("anything_else")
        assert path == "./store/"
        assert (tmp_path / "store").is_dir()

    def test_with_config_fig_dir_uses_root_and_name(self):
        u = Util("test")
        path = u.get_path("fig_dir")
        assert "testexp" in path
        assert "images" in path

    def test_with_config_res_dir_uses_root_and_name(self):
        u = Util("test")
        path = u.get_path("res_dir")
        assert "testexp" in path
        assert "results" in path

    def test_with_config_unknown_key_defaults_to_store(self):
        u = Util("test")
        path = u.get_path("unknown_key")
        assert "store" in path

    def test_get_path_creates_directory(self, tmp_path):
        glob_conf.config["EXP"]["root"] = str(tmp_path)
        glob_conf.config["EXP"]["name"] = "newexp"
        u = Util("test")
        import os

        path = u.get_path("res_dir")
        assert os.path.isdir(path)


# ---------------------------------------------------------------------------
# check_class_label()
# ---------------------------------------------------------------------------


class TestCheckClassLabel:
    def test_renames_class_label_to_target(self):
        u = Util("test")
        import pandas as pd

        df = pd.DataFrame({"emotion": [1, 2], "class_label": ["A", "B"]})
        result = u.check_class_label(df)
        assert "class_label" not in result.columns
        assert "emotion" in result.columns
        # class_label becomes emotion (old emotion column dropped)
        assert list(result["emotion"]) == ["A", "B"]

    def test_no_class_label_column_unchanged(self):
        u = Util("test")
        import pandas as pd

        df = pd.DataFrame({"emotion": [1, 2, 3], "other": [4, 5, 6]})
        result = u.check_class_label(df)
        assert list(result.columns) == ["emotion", "other"]

    def test_no_target_key_in_config_unchanged(self):
        # Remove the target key so config_val returns None (the default)
        del glob_conf.config["DATA"]["target"]
        u = Util("test")
        import pandas as pd

        df = pd.DataFrame({"emotion": [1], "class_label": ["A"]})
        result = u.check_class_label(df)
        # target is None → condition is False, no rename
        assert "class_label" in result.columns


# ---------------------------------------------------------------------------
# config_val_data()
# ---------------------------------------------------------------------------


class TestConfigValData:
    def test_returns_value_from_main_config(self):
        glob_conf.config["DATA"]["emodb.db_path"] = "/some/path"
        u = Util("test")
        result = u.config_val_data("emodb", "db_path", "default")
        assert result == "/some/path"

    def test_returns_default_when_key_missing(self):
        u = Util("test")
        result = u.config_val_data("emodb", "nonexistent", "fallback")
        assert result == "fallback"

    def test_strips_quotes_from_value(self):
        glob_conf.config["DATA"]["emodb.db_path"] = "'quoted/path'"
        u = Util("test")
        result = u.config_val_data("emodb", "db_path", "default")
        assert result == "quoted/path"

    def test_empty_key_looks_up_dataset_directly(self):
        glob_conf.config["DATA"]["emodb"] = "/direct/path"
        u = Util("test")
        result = u.config_val_data("emodb", "", "fallback")
        assert result == "/direct/path"


# ---------------------------------------------------------------------------
# append_to_result_file
# ---------------------------------------------------------------------------


class TestAppendToResultFile:
    def test_creates_file_and_writes_line(self, tmp_path):
        u = Util("test")
        path = str(tmp_path / "results.txt")
        u.append_to_result_file(path, "hello")
        assert (tmp_path / "results.txt").read_text() == "hello\n"

    def test_appends_multiple_lines(self, tmp_path):
        u = Util("test")
        path = str(tmp_path / "results.txt")
        u.append_to_result_file(path, "line1")
        u.append_to_result_file(path, "line2")
        lines = (tmp_path / "results.txt").read_text().splitlines()
        assert lines == ["line1", "line2"]

    def test_appends_to_existing_content(self, tmp_path):
        p = tmp_path / "results.txt"
        p.write_text("existing\n")
        u = Util("test")
        u.append_to_result_file(str(p), "new")
        lines = p.read_text().splitlines()
        assert lines == ["existing", "new"]

    def test_does_not_duplicate_existing_line(self, tmp_path):
        p = tmp_path / "results.txt"
        p.write_text("already here\n")
        u = Util("test")
        u.append_to_result_file(str(p), "already here")
        assert p.read_text() == "already here\n"

    def test_duplicate_check_is_exact_match(self, tmp_path):
        u = Util("test")
        path = str(tmp_path / "results.txt")
        u.append_to_result_file(path, "line")
        u.append_to_result_file(path, "line extra")
        lines = (tmp_path / "results.txt").read_text().splitlines()
        assert lines == ["line", "line extra"]
