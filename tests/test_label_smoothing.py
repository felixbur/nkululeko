"""Unit tests for label_smoothing config parsing in the base Model class."""

import configparser
import tempfile

import pytest
import torch

import nkululeko.glob_conf as glob_conf
from nkululeko.models.model import Model
from nkululeko.utils.util import Util


@pytest.fixture(autouse=True)
def _restore_glob_conf():
    """Save and restore glob_conf state to prevent test leakage."""
    orig_config = getattr(glob_conf, "config", None)
    orig_labels = getattr(glob_conf, "labels", None)
    yield
    glob_conf.config = orig_config
    glob_conf.labels = orig_labels


def _make_config(label_smoothing=None, loss="cross", tmp_dir=None):
    """Helper to create a minimal config for model tests."""
    config = configparser.ConfigParser()
    model_section = {"loss": loss, "layers": "[64]"}
    if label_smoothing is not None:
        model_section["label_smoothing"] = str(label_smoothing)
    config["MODEL"] = model_section
    config["DATA"] = {"target": "emotion"}
    root = tmp_dir if tmp_dir else tempfile.mkdtemp(prefix="nkululeko_test_")
    config["EXP"] = {"root": root, "name": "test_ls"}
    return config


def _make_model_stub(label_smoothing=None, tmp_dir=None):
    """Return a Model instance with only the attributes needed by _get_label_smoothing.

    Uses ``object.__new__`` to bypass ``Model.__init__``, which requires
    train/test dataframes that are unnecessary for this unit test.
    """
    config = _make_config(label_smoothing=label_smoothing, tmp_dir=tmp_dir)
    glob_conf.config = config
    glob_conf.labels = ["anger", "neutral", "happy"]
    model = object.__new__(Model)
    model.util = Util("test")
    return model


class TestLabelSmoothingParsing:
    """Test the _get_label_smoothing helper defined on the base Model class."""

    def test_default_no_smoothing(self, tmp_path):
        model = _make_model_stub(tmp_dir=str(tmp_path))
        assert model._get_label_smoothing() == pytest.approx(0.0)

    def test_true_returns_01(self, tmp_path):
        model = _make_model_stub(label_smoothing="True", tmp_dir=str(tmp_path))
        assert model._get_label_smoothing() == pytest.approx(0.1)

    def test_lowercase_true_returns_01(self, tmp_path):
        model = _make_model_stub(label_smoothing="true", tmp_dir=str(tmp_path))
        assert model._get_label_smoothing() == pytest.approx(0.1)

    def test_string_one_returns_numeric(self, tmp_path):
        """String '1' is parsed as numeric 1.0, which is valid smoothing."""
        model = _make_model_stub(label_smoothing="1", tmp_dir=str(tmp_path))
        assert model._get_label_smoothing() == pytest.approx(1.0)

    def test_boolean_true_returns_01(self, tmp_path):
        model = _make_model_stub(label_smoothing=True, tmp_dir=str(tmp_path))
        assert model._get_label_smoothing() == pytest.approx(0.1)

    def test_float_value_passed_through(self, tmp_path):
        model = _make_model_stub(label_smoothing="0.2", tmp_dir=str(tmp_path))
        assert model._get_label_smoothing() == pytest.approx(0.2)

    def test_zero_returns_zero(self, tmp_path):
        model = _make_model_stub(label_smoothing="0.0", tmp_dir=str(tmp_path))
        assert model._get_label_smoothing() == pytest.approx(0.0)

    def test_false_returns_zero(self, tmp_path):
        model = _make_model_stub(label_smoothing="False", tmp_dir=str(tmp_path))
        assert model._get_label_smoothing() == pytest.approx(0.0)

    def test_invalid_string_returns_zero(self, tmp_path):
        """Non-parsable values should fall back to 0.0 with a warning."""
        model = _make_model_stub(label_smoothing="foo", tmp_dir=str(tmp_path))
        assert model._get_label_smoothing() == pytest.approx(0.0)

    def test_negative_value_returns_zero(self, tmp_path):
        """Negative values are out of range and should fall back to 0.0."""
        model = _make_model_stub(label_smoothing="-0.1", tmp_dir=str(tmp_path))
        assert model._get_label_smoothing() == pytest.approx(0.0)

    def test_value_above_one_returns_zero(self, tmp_path):
        """Values > 1.0 are out of range and should fall back to 0.0."""
        model = _make_model_stub(label_smoothing="2.0", tmp_dir=str(tmp_path))
        assert model._get_label_smoothing() == pytest.approx(0.0)


class TestCrossEntropyWithLabelSmoothing:
    """Verify that torch.nn.CrossEntropyLoss uses label_smoothing correctly."""

    def test_no_smoothing_default(self):
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
        logits = torch.tensor([[10.0, 0.0, 0.0]])
        target = torch.tensor([0])
        loss = loss_fn(logits, target)
        assert loss.item() < 0.01

    def test_smoothing_increases_loss_on_confident_correct(self):
        """Label smoothing should increase loss for over-confident correct predictions."""
        logits = torch.tensor([[10.0, 0.0, 0.0]])
        target = torch.tensor([0])
        loss_no_smooth = torch.nn.CrossEntropyLoss(label_smoothing=0.0)(logits, target)
        loss_smooth = torch.nn.CrossEntropyLoss(label_smoothing=0.1)(logits, target)
        assert loss_smooth.item() > loss_no_smooth.item()

    def test_smoothing_on_confident_incorrect_prediction(self):
        """Label smoothing should decrease loss for highly confident incorrect predictions."""
        logits = torch.tensor([[0.0, 10.0, 0.0]])
        target = torch.tensor([0])
        loss_no_smooth = torch.nn.CrossEntropyLoss(label_smoothing=0.0)(logits, target)
        loss_smooth = torch.nn.CrossEntropyLoss(label_smoothing=0.1)(logits, target)
        assert loss_smooth.item() < loss_no_smooth.item()


class TestSetupCriterionIntegration:
    """Verify that _setup_criterion wires label_smoothing into CrossEntropyLoss."""

    def test_criterion_has_label_smoothing(self, tmp_path):
        """Model.criterion should receive the parsed label_smoothing value."""
        model = _make_model_stub(label_smoothing="0.15", tmp_dir=str(tmp_path))
        model.class_num = 3
        model._setup_criterion()
        assert hasattr(model, "criterion")
        assert model.criterion.label_smoothing == pytest.approx(0.15)

    def test_criterion_zero_smoothing_by_default(self, tmp_path):
        """Without label_smoothing config, criterion should use 0.0."""
        model = _make_model_stub(tmp_dir=str(tmp_path))
        model.class_num = 3
        model._setup_criterion()
        assert model.criterion.label_smoothing == pytest.approx(0.0)
