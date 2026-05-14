"""Unit tests for ONNX export in model.py (export_onnx).

audplot and other heavy optional dependencies are stubbed out at the module
level so this file can run in environments where they are not installed.
"""

import configparser
import importlib
import sys
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

import nkululeko.glob_conf as glob_conf


@pytest.fixture(autouse=True)
def setup_glob_conf(tmp_path):
    config = configparser.ConfigParser()
    config["EXP"] = {"type": "classification", "name": "testexp", "root": str(tmp_path)}
    config["DATA"] = {"target": "emotion", "databases": "['emodb']"}
    config["MODEL"] = {"type": "svm", "n_jobs": "1"}
    config["FEATS"] = {"type": "['os']"}
    glob_conf.config = config
    yield
    glob_conf.config = None


@pytest.fixture
def model_class(monkeypatch):
    original_modules = {
        "nkululeko.reporting.reporter": sys.modules.get("nkululeko.reporting.reporter"),
        "nkululeko.models.model": sys.modules.get("nkululeko.models.model"),
    }

    for module_name in ("audplot", "audmetric", "seaborn", "confidence_intervals"):
        monkeypatch.setitem(sys.modules, module_name, MagicMock())

    for module_name in original_modules:
        sys.modules.pop(module_name, None)

    model_module = importlib.import_module("nkululeko.models.model")
    yield model_module.Model

    for module_name, original_module in original_modules.items():
        if original_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module


@pytest.fixture
def trained_model(model_class):
    """Model fixture with a mock sklearn classifier attached."""
    rng = np.random.default_rng(42)
    df_train = pd.DataFrame({"emotion": ["happy", "sad", "angry", "happy"]})
    df_test = pd.DataFrame({"emotion": ["sad", "angry"]})
    feats_train = pd.DataFrame(rng.random((4, 5)))
    feats_test = pd.DataFrame(rng.random((2, 5)))
    model = model_class(df_train, df_test, feats_train, feats_test)
    model.clf = MagicMock()
    return model


def _onnx_mocks():
    """Return a dict suitable for patch.dict(sys.modules, ...) that stubs skl2onnx."""
    fake_onnx_model = MagicMock()
    fake_onnx_model.SerializeToString.return_value = b"onnx_bytes"

    fake_skl2onnx = MagicMock()
    fake_skl2onnx.convert_sklearn = MagicMock(return_value=fake_onnx_model)

    fake_data_types = MagicMock()
    fake_data_types.FloatTensorType = MagicMock(
        side_effect=lambda shape: ("FTT", shape)
    )

    fake_common = MagicMock()
    fake_common.data_types = fake_data_types

    return (
        {
            "skl2onnx": fake_skl2onnx,
            "skl2onnx.common": fake_common,
            "skl2onnx.common.data_types": fake_data_types,
        },
        fake_skl2onnx,
        fake_onnx_model,
    )


class TestExportOnnx:
    def test_export_writes_serialized_bytes(self, trained_model, tmp_path):
        """export_onnx writes the serialized ONNX bytes to the given path."""
        onnx_path = str(tmp_path / "model.onnx")
        mocks, _, fake_onnx_model = _onnx_mocks()
        fake_onnx_model.SerializeToString.return_value = b"onnx_bytes"

        with patch.dict(sys.modules, mocks):
            with patch("builtins.open", mock_open()) as mocked_file:
                trained_model.export_onnx(onnx_path)
                handle = mocked_file()
                handle.write.assert_called_once_with(b"onnx_bytes")

    def test_export_infers_n_features_from_feats_train(self, trained_model, tmp_path):
        """When input_shape is None, n_features is inferred from feats_train shape."""
        onnx_path = str(tmp_path / "model.onnx")
        captured = {}

        mocks, fake_skl2onnx, _ = _onnx_mocks()

        def capture_convert(clf, initial_types):
            captured["initial_types"] = initial_types
            m = MagicMock()
            m.SerializeToString.return_value = b""
            return m

        fake_skl2onnx.convert_sklearn = capture_convert

        with patch.dict(sys.modules, mocks):
            with patch("builtins.open", mock_open()):
                trained_model.export_onnx(onnx_path)

        # initial_types[0] is ("input", FloatTensorType([None, n_features]))
        # FloatTensorType mock returns ("FTT", shape) where shape is [None, 5]
        _, ftt_result = captured["initial_types"][0]
        # ftt_result is ("FTT", [None, 5]) — second element is the shape arg
        shape_arg = ftt_result[1]
        assert 5 in shape_arg  # feats_train has 5 columns

    def test_export_uses_explicit_input_shape(self, trained_model, tmp_path):
        """When input_shape is provided, it is forwarded to convert_sklearn."""
        onnx_path = str(tmp_path / "model.onnx")
        explicit_shape = [None, 10]
        captured = {}

        mocks, fake_skl2onnx, _ = _onnx_mocks()

        def capture_convert(clf, initial_types):
            captured["initial_types"] = initial_types
            m = MagicMock()
            m.SerializeToString.return_value = b""
            return m

        fake_skl2onnx.convert_sklearn = capture_convert

        with patch.dict(sys.modules, mocks):
            with patch("builtins.open", mock_open()):
                trained_model.export_onnx(onnx_path, input_shape=explicit_shape)

        _, ftt_result = captured["initial_types"][0]
        # FloatTensorType was called with explicit_shape → ftt_result[1] == explicit_shape
        assert ftt_result[1] == explicit_shape

    def test_export_no_clf_calls_error_and_returns(self, trained_model, tmp_path):
        """export_onnx calls util.error when clf is not set.

        skl2onnx must be mocked because the import occurs before the hasattr check.
        """
        del trained_model.clf
        onnx_path = str(tmp_path / "model.onnx")

        error_msgs = []

        def capture_error(msg):
            error_msgs.append(msg)
            raise SystemExit(1)

        trained_model.util.error = capture_error

        mocks, _, _ = _onnx_mocks()
        with patch.dict(sys.modules, mocks):
            with pytest.raises(SystemExit):
                trained_model.export_onnx(onnx_path)

        assert len(error_msgs) == 1
        assert "No trained model" in error_msgs[0]

    def test_export_calls_convert_sklearn_with_clf(self, trained_model, tmp_path):
        """convert_sklearn is called with the model's clf object."""
        onnx_path = str(tmp_path / "model.onnx")
        captured_clf = {}

        mocks, fake_skl2onnx, _ = _onnx_mocks()

        def capture_convert(clf, initial_types):
            captured_clf["clf"] = clf
            m = MagicMock()
            m.SerializeToString.return_value = b""
            return m

        fake_skl2onnx.convert_sklearn = capture_convert

        with patch.dict(sys.modules, mocks):
            with patch("builtins.open", mock_open()):
                trained_model.export_onnx(onnx_path)

        assert captured_clf["clf"] is trained_model.clf
