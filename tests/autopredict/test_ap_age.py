from unittest.mock import Mock, patch

import pandas as pd

from nkululeko.autopredict.ap_age import AgePredictor
from nkululeko.experiment_context import ExperimentContext, use_context


class TestAgePredictor:
    def test_init(self):
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = AgePredictor(df)

        assert predictor.df is df
        assert predictor.util is not None

    @patch("nkululeko.autopredict.ap_age.FeatureExtractor")
    def test_predict(self, mock_feature_extractor):
        context = ExperimentContext(config={"DATA": {"databases": "['test_db']"}})

        # Create mock dataframe with age values
        mock_age_df = pd.DataFrame({"age": [0.25, 0.45, 0.70]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_age_df
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1, 2, 3]})
        with use_context(context):
            predictor = AgePredictor(df)
            result = predictor.predict("train")

        assert "age_pred" in result.columns
        assert len(result) == 3
        assert result["age_pred"].dtype == int
        # Age is multiplied by 100 and cast to int
        assert list(result["age_pred"]) == [25, 45, 70]
