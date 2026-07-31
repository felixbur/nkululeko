from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from nkululeko.autopredict.ap_pesq import PESQPredictor
from nkululeko.experiment_context import ExperimentContext, use_context


class TestPESQPredictor:
    def test_init(self):
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = PESQPredictor(df)

        assert predictor.df is df
        assert predictor.util is not None

    @patch("nkululeko.autopredict.ap_pesq.FeatureExtractor")
    def test_predict(self, mock_feature_extractor):
        context = ExperimentContext(config={"DATA": {"databases": "['test_db']"}})

        # Create mock dataframe with PESQ values
        mock_pesq_df = pd.DataFrame({"pesq": [3.5, 4.2, 2.8]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_pesq_df
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1, 2, 3]})
        with use_context(context):
            predictor = PESQPredictor(df)

            result = predictor.predict("train")

        assert "pesq_pred" in result.columns
        assert len(result) == 3
        assert result["pesq_pred"].iloc[0] == pytest.approx(3.5, abs=0.01)

    @patch("nkululeko.autopredict.ap_pesq.FeatureExtractor")
    def test_predict_handles_nan_and_inf(self, mock_feature_extractor):
        context = ExperimentContext(config={"DATA": {"databases": "['test_db']"}})

        mock_pesq_df = pd.DataFrame({"pesq": [3.5, np.nan, np.inf, -np.inf]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_pesq_df
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1, 2, 3, 4]})
        with use_context(context):
            predictor = PESQPredictor(df)

            result = predictor.predict("train")

        assert result["pesq_pred"].iloc[1] == pytest.approx(0.0)
        assert result["pesq_pred"].iloc[2] == pytest.approx(0.0)
        assert result["pesq_pred"].iloc[3] == pytest.approx(0.0)
