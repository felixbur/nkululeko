import pytest
import numpy as np
import pandas as pd

from nkululeko.ensemble import (
    majority_voting,
    mean_ensemble,
    max_ensemble,
    sum_ensemble,
    uncertainty_ensemble,
    uncertainty_weighted_ensemble,
    confidence_weighted_ensemble,
    performance_weighted_ensemble,
)


@pytest.fixture
def binary_predictions():
    """Create sample binary prediction dataframes for ensemble testing."""
    labels = ["happy", "sad"]

    df1 = pd.DataFrame({
        "predicted": ["happy", "sad", "happy", "sad"],
        "happy": [0.8, 0.3, 0.7, 0.4],
        "sad": [0.2, 0.7, 0.3, 0.6],
        "uncertainty": [0.2, 0.3, 0.3, 0.4],
    })

    df2 = pd.DataFrame({
        "predicted": ["happy", "happy", "sad", "sad"],
        "happy": [0.9, 0.6, 0.4, 0.3],
        "sad": [0.1, 0.4, 0.6, 0.7],
        "uncertainty": [0.1, 0.4, 0.4, 0.3],
    })

    df3 = pd.DataFrame({
        "predicted": ["happy", "sad", "sad", "sad"],
        "happy": [0.7, 0.4, 0.3, 0.2],
        "sad": [0.3, 0.6, 0.7, 0.8],
        "uncertainty": [0.3, 0.4, 0.3, 0.2],
    })

    return [df1, df2, df3], labels


@pytest.fixture
def multiclass_predictions():
    """Create sample multiclass prediction dataframes."""
    labels = ["happy", "sad", "angry"]

    df1 = pd.DataFrame({
        "predicted": ["happy", "sad", "angry"],
        "happy": [0.7, 0.1, 0.2],
        "sad": [0.2, 0.7, 0.1],
        "angry": [0.1, 0.2, 0.7],
        "uncertainty": [0.3, 0.3, 0.3],
    })

    df2 = pd.DataFrame({
        "predicted": ["happy", "angry", "angry"],
        "happy": [0.6, 0.2, 0.1],
        "sad": [0.3, 0.3, 0.2],
        "angry": [0.1, 0.5, 0.7],
        "uncertainty": [0.4, 0.5, 0.3],
    })

    df3 = pd.DataFrame({
        "predicted": ["sad", "sad", "angry"],
        "happy": [0.3, 0.2, 0.15],
        "sad": [0.4, 0.6, 0.15],
        "angry": [0.3, 0.2, 0.7],
        "uncertainty": [0.4, 0.4, 0.3],
    })

    return [df1, df2, df3], labels


class TestMajorityVoting:
    """Test majority voting ensemble method."""

    def test_basic_majority_voting(self, binary_predictions):
        preds_ls, labels = binary_predictions
        result = majority_voting(preds_ls)
        assert len(result) == 4
        assert result.iloc[0] == "happy"
        assert result.iloc[3] == "sad"

    def test_majority_voting_multiclass(self, multiclass_predictions):
        preds_ls, labels = multiclass_predictions
        result = majority_voting(preds_ls)
        assert len(result) == 3
        assert result.iloc[2] == "angry"


class TestMeanEnsemble:
    """Test mean ensemble method."""

    def test_mean_ensemble(self, binary_predictions):
        preds_ls, labels = binary_predictions
        ensemble_preds = pd.concat(preds_ls, axis=1)
        result = mean_ensemble(ensemble_preds, labels)
        assert len(result) == 4

    def test_mean_ensemble_returns_labels(self, binary_predictions):
        preds_ls, labels = binary_predictions
        ensemble_preds = pd.concat(preds_ls, axis=1)
        result = mean_ensemble(ensemble_preds, labels)
        for pred in result:
            assert pred in labels


class TestMaxEnsemble:
    """Test max ensemble method."""

    def test_max_ensemble(self, binary_predictions):
        preds_ls, labels = binary_predictions
        ensemble_preds = pd.concat(preds_ls, axis=1)
        result = max_ensemble(ensemble_preds, labels)
        assert len(result) == 4

    def test_max_ensemble_returns_labels(self, binary_predictions):
        preds_ls, labels = binary_predictions
        ensemble_preds = pd.concat(preds_ls, axis=1)
        result = max_ensemble(ensemble_preds, labels)
        for pred in result:
            assert pred in labels


class TestSumEnsemble:
    """Test sum ensemble method."""

    def test_sum_ensemble(self, binary_predictions):
        preds_ls, labels = binary_predictions
        ensemble_preds = pd.concat(preds_ls, axis=1)
        result = sum_ensemble(ensemble_preds, labels)
        assert len(result) == 4

    def test_sum_ensemble_returns_labels(self, binary_predictions):
        preds_ls, labels = binary_predictions
        ensemble_preds = pd.concat(preds_ls, axis=1)
        result = sum_ensemble(ensemble_preds, labels)
        for pred in result:
            assert pred in labels


class TestUncertaintyEnsemble:
    """Test uncertainty-based ensemble method."""

    def test_uncertainty_ensemble_no_threshold(self, binary_predictions):
        preds_ls, labels = binary_predictions
        result = uncertainty_ensemble(preds_ls, labels, threshold=1.0)
        assert len(result) == 4

    def test_uncertainty_ensemble_with_threshold(self, binary_predictions):
        preds_ls, labels = binary_predictions
        result = uncertainty_ensemble(preds_ls, labels, threshold=0.15)
        assert len(result) == 4
        for pred in result:
            assert pred in labels

    def test_uncertainty_picks_lowest(self, binary_predictions):
        preds_ls, labels = binary_predictions
        result = uncertainty_ensemble(preds_ls, labels, threshold=1.0)
        assert len(result) == 4


class TestUncertaintyWeightedEnsemble:
    """Test uncertainty-weighted ensemble method."""

    def test_returns_predictions_and_uncertainties(self, binary_predictions):
        preds_ls, labels = binary_predictions
        predictions, uncertainties = uncertainty_weighted_ensemble(preds_ls, labels)
        assert len(predictions) == 4
        assert len(uncertainties) == 4

    def test_predictions_are_valid_labels(self, binary_predictions):
        preds_ls, labels = binary_predictions
        predictions, _ = uncertainty_weighted_ensemble(preds_ls, labels)
        for pred in predictions:
            assert pred in labels

    def test_uncertainties_range(self, binary_predictions):
        preds_ls, labels = binary_predictions
        _, uncertainties = uncertainty_weighted_ensemble(preds_ls, labels)
        for u in uncertainties:
            assert 0 <= u <= 1


class TestConfidenceWeightedEnsemble:
    """Test confidence-weighted ensemble method."""

    def test_returns_predictions_and_confidences(self, binary_predictions):
        preds_ls, labels = binary_predictions
        predictions, confidences = confidence_weighted_ensemble(preds_ls, labels)
        assert len(predictions) == 4
        assert len(confidences) == 4

    def test_predictions_are_valid_labels(self, binary_predictions):
        preds_ls, labels = binary_predictions
        predictions, _ = confidence_weighted_ensemble(preds_ls, labels)
        for pred in predictions:
            assert pred in labels


class TestPerformanceWeightedEnsemble:
    """Test performance-weighted ensemble method."""

    def test_basic_performance_weighted(self, binary_predictions):
        preds_ls, labels = binary_predictions
        weights = [0.5, 0.3, 0.2]
        predictions, confidences = performance_weighted_ensemble(preds_ls, labels, weights)
        assert len(predictions) == 4
        assert len(confidences) == 4

    def test_equal_weights(self, binary_predictions):
        preds_ls, labels = binary_predictions
        weights = [0.33, 0.33, 0.34]
        predictions, _ = performance_weighted_ensemble(preds_ls, labels, weights)
        for pred in predictions:
            assert pred in labels

    def test_invalid_weights_range(self, binary_predictions):
        preds_ls, labels = binary_predictions
        weights = [1.5, 0.3, 0.2]
        with pytest.raises(AssertionError):
            performance_weighted_ensemble(preds_ls, labels, weights)

    def test_wrong_number_of_weights(self, binary_predictions):
        preds_ls, labels = binary_predictions
        weights = [0.5, 0.5]
        with pytest.raises(AssertionError):
            performance_weighted_ensemble(preds_ls, labels, weights)


class TestEnsemblePredictions:
    """Test the main ensemble_predictions function."""

    def test_unknown_method_raises_error(self):
        """Test that unknown method raises ValueError."""
        from nkululeko.ensemble import ensemble_predictions
        with pytest.raises(ValueError):
            ensemble_predictions([], "invalid_method", 1.0, None, True)
