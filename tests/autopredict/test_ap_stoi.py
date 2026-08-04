from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from nkululeko.autopredict.ap_stoi import STOIPredictor
from nkululeko.experiment_context import ExperimentContext, use_context


class TestSTOIPredictor:
    def test_init(self):
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = STOIPredictor(df)

        assert predictor.df is df
        assert predictor.util is not None

    @patch("nkululeko.autopredict.ap_stoi.FeatureExtractor")
    def test_predict(self, mock_feature_extractor):
        context = ExperimentContext(config={"DATA": {"databases": "['test_db']"}})

        # Create mock dataframe with STOI values
        mock_stoi_df = pd.DataFrame({"stoi": [0.85, 0.92, 0.78]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_stoi_df
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1, 2, 3]})
        with use_context(context):
            predictor = STOIPredictor(df)

            result = predictor.predict("train")

        assert "stoi_pred" in result.columns
        assert len(result) == 3
        assert result["stoi_pred"].iloc[0] == pytest.approx(0.85, abs=0.01)

    @patch("nkululeko.autopredict.ap_stoi.FeatureExtractor")
    def test_predict_handles_nan_and_inf(self, mock_feature_extractor):
        context = ExperimentContext(config={"DATA": {"databases": "['test_db']"}})

        mock_stoi_df = pd.DataFrame({"stoi": [0.85, np.nan, np.inf, -np.inf]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_stoi_df
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1, 2, 3, 4]})
        with use_context(context):
            predictor = STOIPredictor(df)

            result = predictor.predict("train")

        assert result["stoi_pred"].iloc[1] == pytest.approx(0.0)
        assert result["stoi_pred"].iloc[2] == pytest.approx(0.0)
        assert result["stoi_pred"].iloc[3] == pytest.approx(0.0)
