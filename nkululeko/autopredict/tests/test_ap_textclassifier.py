from unittest.mock import Mock, patch

import pandas as pd
import pytest

from nkululeko.autopredict.ap_textclassifier import TextClassificationPredictor


class TestTextClassificationPredictor:
    def test_init_with_util(self):
        mock_util = Mock()
        df = pd.DataFrame({"text": ["hello", "world"]})
        
        predictor = TextClassificationPredictor(df, util=mock_util)
        
        assert predictor.df is df
        assert predictor.util is mock_util

    @patch("nkululeko.autopredict.ap_textclassifier.Util")
    def test_init_without_util(self, mock_util_class):
        mock_util = Mock()
        mock_util_class.return_value = mock_util
        
        df = pd.DataFrame({"text": ["hello", "world"]})
        predictor = TextClassificationPredictor(df)
        
        assert predictor.df is df
        mock_util_class.assert_called_once_with("textClassifierPredictor")

    @patch("nkululeko.autopredict.ap_textclassifier.TextClassifier")
    def test_predict(self, mock_text_classifier_class):
        mock_util = Mock()
        
        # Create mock classifier
        mock_classifier = Mock()
        mock_result_df = pd.DataFrame({
            "category": ["positive", "negative"]
        })
        mock_classifier.extract.return_value = mock_result_df
        mock_text_classifier_class.return_value = mock_classifier
        
        df = pd.DataFrame({"text": ["hello", "world"]})
        predictor = TextClassificationPredictor(df, util=mock_util)
        
        result = predictor.predict("train")
        
        assert "text" in result.columns
        assert "category" in result.columns
        assert len(result) == 2
        mock_classifier.extract.assert_called_once()
