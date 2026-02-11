import pytest
import os
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.preprocessing import LabelEncoder
import nkululeko.glob_conf as glob_conf


@pytest.fixture
def setup_glob_conf():
    """Set up glob_conf for testing."""
    config = {
        "DATA": {"target": "emotion"},
        "FEATS": {},
        "MODEL": {"type": "xgb"},
    }
    original_config = getattr(glob_conf, "config", None)
    glob_conf.config = config
    yield config
    if original_config is not None:
        glob_conf.config = original_config
    else:
        glob_conf.config = None


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.get_predictions.return_value = (np.array([0, 1, 0, 1]), None)
    model.predict.return_value = MagicMock(
        result=MagicMock(get_result=MagicMock(return_value=0.85))
    )
    return model


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    df = pd.DataFrame({
        "emotion": [0, 1, 0, 1],
        "speaker": ["s1", "s2", "s3", "s4"],
        "class_label": ["happy", "sad", "happy", "sad"],
    })
    df.is_labeled = True
    return df


@pytest.fixture
def label_encoder():
    """Create a fitted label encoder."""
    le = LabelEncoder()
    le.fit(["happy", "sad"])
    return le


class TestTestPredictorInit:
    """Test TestPredictor initialization."""

    def test_initialization(self, mock_model, sample_dataframe, label_encoder, setup_glob_conf):
        from nkululeko.testing_predictor import TestPredictor
        predictor = TestPredictor(mock_model, sample_dataframe, label_encoder, "test_results.csv")
        assert predictor.model is mock_model
        assert predictor.orig_df is sample_dataframe
        assert predictor.label_encoder is label_encoder
        assert predictor.target == "emotion"
        assert predictor.name == "test_results.csv"


class TestTestPredictorPredict:
    """Test predict_and_store method."""

    def test_predict_basic_path(self, mock_model, sample_dataframe, label_encoder, setup_glob_conf):
        """Test the basic prediction path (no label_data, no tests)."""
        from nkululeko.testing_predictor import TestPredictor

        tmpfile = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
        predictor = TestPredictor(mock_model, sample_dataframe, label_encoder, tmpfile)

        result = predictor.predict_and_store()

        assert result == 0
        assert os.path.isfile(tmpfile)

        saved_df = pd.read_csv(tmpfile, index_col=0)
        assert "predictions" in saved_df.columns
        os.remove(tmpfile)

    def test_predict_with_class_label_rename(self, label_encoder, setup_glob_conf):
        """Test that class_label column is properly renamed."""
        df = pd.DataFrame({
            "emotion": [0, 1],
            "class_label": ["happy", "sad"],
            "speaker": ["s1", "s2"],
        })
        df.is_labeled = True

        model = MagicMock()
        model.get_predictions.return_value = (np.array([0, 1]), None)

        tmpfile = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
        from nkululeko.testing_predictor import TestPredictor
        predictor = TestPredictor(model, df, label_encoder, tmpfile)

        predictor.predict_and_store()

        saved_df = pd.read_csv(tmpfile, index_col=0)
        assert "emotion" in saved_df.columns
        assert "class_label" not in saved_df.columns
        os.remove(tmpfile)


class TestTestPredictorEdgeCases:
    """Test edge cases."""

    def test_label_encoder_inverse_transform(self, label_encoder):
        """Test label encoder handles predictions correctly."""
        predictions = np.array([0, 1, 0, 1])
        decoded = label_encoder.inverse_transform(predictions)
        assert list(decoded) == ["happy", "sad", "happy", "sad"]

    def test_empty_predictions(self, label_encoder):
        """Test handling of empty predictions."""
        predictions = np.array([], dtype=int)
        decoded = label_encoder.inverse_transform(predictions)
        assert len(decoded) == 0
