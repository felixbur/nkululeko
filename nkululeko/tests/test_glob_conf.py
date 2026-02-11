import pytest
import nkululeko.glob_conf as glob_conf


class TestGlobConf:
    """Test global configuration management."""

    def setup_method(self):
        """Reset all globals before each test."""
        glob_conf.config = None
        glob_conf.label_encoder = None
        glob_conf.util = None
        glob_conf.module = None
        glob_conf.report = None
        glob_conf.labels = None
        glob_conf.target = None

    def test_initial_state(self):
        """Test that all globals start as None."""
        assert glob_conf.config is None
        assert glob_conf.label_encoder is None
        assert glob_conf.util is None
        assert glob_conf.module is None
        assert glob_conf.report is None
        assert glob_conf.labels is None
        assert glob_conf.target is None

    def test_init_config(self):
        """Test init_config sets config global."""
        mock_config = {"EXP": {"name": "test"}}
        glob_conf.init_config(mock_config)
        assert glob_conf.config == mock_config
        assert glob_conf.config["EXP"]["name"] == "test"

    def test_init_config_overwrite(self):
        """Test init_config overwrites existing config."""
        glob_conf.init_config({"first": True})
        glob_conf.init_config({"second": True})
        assert "second" in glob_conf.config
        assert "first" not in glob_conf.config

    def test_set_label_encoder(self):
        """Test set_label_encoder sets label_encoder global."""
        mock_encoder = type("MockEncoder", (), {"transform": lambda self, x: x})()
        glob_conf.set_label_encoder(mock_encoder)
        assert glob_conf.label_encoder is mock_encoder

    def test_set_util(self):
        """Test set_util sets util global."""
        mock_util = type("MockUtil", (), {"debug": lambda self, msg: None})()
        glob_conf.set_util(mock_util)
        assert glob_conf.util is mock_util

    def test_set_module(self):
        """Test set_module sets module global."""
        glob_conf.set_module("nkululeko")
        assert glob_conf.module == "nkululeko"

    def test_set_module_different_values(self):
        """Test set_module with various module names."""
        for module_name in ["demo", "predict", "ensemble", "flags"]:
            glob_conf.set_module(module_name)
            assert glob_conf.module == module_name

    def test_set_report(self):
        """Test set_report sets report global."""
        mock_report = {"result": 0.85}
        glob_conf.set_report(mock_report)
        assert glob_conf.report == mock_report

    def test_set_labels(self):
        """Test set_labels sets labels global."""
        labels = ["happy", "sad", "angry"]
        glob_conf.set_labels(labels)
        assert glob_conf.labels == labels
        assert len(glob_conf.labels) == 3

    def test_set_labels_empty(self):
        """Test set_labels with empty list."""
        glob_conf.set_labels([])
        assert glob_conf.labels == []

    def test_set_target(self):
        """Test set_target sets target global."""
        glob_conf.set_target("emotion")
        assert glob_conf.target == "emotion"

    def test_set_target_different_values(self):
        """Test set_target with various target names."""
        for target in ["emotion", "valence", "arousal", "dominance"]:
            glob_conf.set_target(target)
            assert glob_conf.target == target

    def test_all_setters_together(self):
        """Test all setters work together without interference."""
        glob_conf.init_config({"EXP": {"name": "test"}})
        glob_conf.set_label_encoder("encoder")
        glob_conf.set_util("util")
        glob_conf.set_module("test_module")
        glob_conf.set_report("report")
        glob_conf.set_labels(["a", "b"])
        glob_conf.set_target("emotion")

        assert glob_conf.config == {"EXP": {"name": "test"}}
        assert glob_conf.label_encoder == "encoder"
        assert glob_conf.util == "util"
        assert glob_conf.module == "test_module"
        assert glob_conf.report == "report"
        assert glob_conf.labels == ["a", "b"]
        assert glob_conf.target == "emotion"

    def test_set_none_values(self):
        """Test that setters accept None."""
        glob_conf.init_config(None)
        glob_conf.set_label_encoder(None)
        glob_conf.set_util(None)
        glob_conf.set_module(None)
        glob_conf.set_report(None)
        glob_conf.set_labels(None)
        glob_conf.set_target(None)

        assert glob_conf.config is None
        assert glob_conf.label_encoder is None
        assert glob_conf.util is None
        assert glob_conf.module is None
        assert glob_conf.report is None
        assert glob_conf.labels is None
        assert glob_conf.target is None
