from unittest.mock import Mock, patch

import pandas as pd
import pytest

from nkululeko.autopredict.ap_dominance import DominancePredictor
from nkululeko.experiment_context import ExperimentContext, use_context


class TestDominancePredictor:
    def test_init(self):
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = DominancePredictor(df)

        assert predictor.df is df
        assert predictor.util is not None

    @patch("nkululeko.autopredict.ap_dominance.FeatureExtractor")
    def test_predict(self, mock_feature_extractor):
        context = ExperimentContext(config={"DATA": {"databases": "['test_db']"}})

        # Create mock dataframe with dominance values
        mock_dominance_df = pd.DataFrame({"dominance": [0.25, 0.45, 0.70]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_dominance_df
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1, 2, 3]})
        with use_context(context):
            predictor = DominancePredictor(df)

            result = predictor.predict("train")

        assert "dominance_pred" in result.columns
        assert len(result) == 3
        # Values should be multiplied by 1000, cast to int, then divided by 1000
        assert result["dominance_pred"].iloc[0] == pytest.approx(0.25, abs=0.001)
