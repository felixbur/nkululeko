from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from nkululeko.autopredict.ap_audmodel import AudmodelPredictor
from nkululeko.experiment_context import ExperimentContext, use_context


def _context(embeddings_name):
    return ExperimentContext(
        config={
            "DATA": {"databases": "['test_db']"},
            "FEATS": {"audmodel.embeddings_name": embeddings_name},
        }
    )


class TestAudmodelPredictor:
    def test_init(self):
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = AudmodelPredictor(df)

        assert predictor.df is df
        assert predictor.util is not None

    @patch("nkululeko.autopredict.ap_audmodel.FeatureExtractor")
    def test_single_dim_head_is_copied_as_is(self, mock_feature_extractor):
        pred_df = pd.DataFrame({"valence": [0.1, 0.5, 0.9]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = pred_df
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1, 2, 3]})
        with use_context(_context("valence")):
            result = AudmodelPredictor(df).predict("train")

        assert list(result["valence_pred"]) == [0.1, 0.5, 0.9]
        # requests the audmodel feats_type, not a fixed/specific one
        args, _ = mock_feature_extractor.call_args
        assert args[1] == ["audmodel"]

    @patch("nkululeko.autopredict.ap_audmodel.FeatureExtractor")
    def test_categorical_head_picks_argmax_and_strips_prefix(
        self, mock_feature_extractor
    ):
        pred_df = pd.DataFrame(
            {
                "emotion.acoustic_angry": [0.7, 0.1, np.nan],
                "emotion.acoustic_happy": [0.2, 0.8, np.nan],
                "emotion.acoustic_neutral": [0.1, 0.1, np.nan],
            }
        )
        mock_extractor = Mock()
        mock_extractor.extract.return_value = pred_df
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1, 2, 3]})
        with use_context(_context("emotion.acoustic")):
            result = AudmodelPredictor(df).predict("train")

        assert list(result["emotion.acoustic_pred"]) == ["angry", "happy", "unknown"]

    @patch("nkululeko.autopredict.ap_audmodel.FeatureExtractor")
    def test_multi_head_writes_one_column_per_head(self, mock_feature_extractor):
        pred_df = pd.DataFrame(
            {
                "valence": [0.1, 0.9],
                "emotion.acoustic_angry": [0.8, 0.1],
                "emotion.acoustic_happy": [0.2, 0.9],
            }
        )
        mock_extractor = Mock()
        mock_extractor.extract.return_value = pred_df
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1, 2]})
        with use_context(_context("['valence', 'emotion.acoustic']")):
            result = AudmodelPredictor(df).predict("train")

        assert list(result["valence_pred"]) == [0.1, 0.9]
        assert list(result["emotion.acoustic_pred"]) == ["angry", "happy"]

    @patch("nkululeko.autopredict.ap_audmodel.FeatureExtractor")
    def test_head_missing_entirely_falls_back_to_unknown(self, mock_feature_extractor):
        """Every row failed extraction (extract() returns no columns at
        all): there's no numeric value to report and no scores to argmax,
        so the column still gets written, as "unknown" for every row,
        rather than being silently omitted."""
        pred_df = pd.DataFrame(index=[0, 1, 2])
        mock_extractor = Mock()
        mock_extractor.extract.return_value = pred_df
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1, 2, 3]})
        with use_context(_context("valence")):
            result = AudmodelPredictor(df).predict("train")

        assert list(result["valence_pred"]) == ["unknown", "unknown", "unknown"]

    @patch("nkululeko.autopredict.ap_audmodel.FeatureExtractor")
    def test_a_row_with_all_nan_scores_is_unknown_not_a_crash(
        self, mock_feature_extractor
    ):
        """Review on #468: with AudmodelSet.extract() now preserving a
        failed file as a NaN row instead of aborting, a categorical head's
        row can be all-NaN rather than all-zero. idxmax(axis=1) on an
        all-NaN row is deprecated in pandas (warns it will raise
        ValueError in a future version) -- must not hit that path, and
        must still end up "unknown" like the all-zero case did."""
        pred_df = pd.DataFrame(
            {
                "emotion.acoustic_angry": [0.7, np.nan],
                "emotion.acoustic_happy": [0.2, np.nan],
            }
        )
        mock_extractor = Mock()
        mock_extractor.extract.return_value = pred_df
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1, 2]})
        with use_context(_context("emotion.acoustic")):
            result = AudmodelPredictor(df).predict("train")

        assert list(result["emotion.acoustic_pred"]) == ["angry", "unknown"]

    @patch("nkululeko.autopredict.ap_audmodel.FeatureExtractor")
    def test_valid_negative_sum_row_is_not_misclassified_as_unknown(
        self, mock_feature_extractor
    ):
        """Review on #468: a generic audmodel head isn't guaranteed to
        output non-negative probabilities -- some models emit raw logits,
        where a legitimately successful row can sum to <= 0 (e.g.
        [-2, 1] or [1, -1]). Failure must be detected from the NaNs
        AudmodelSet.extract() actually produces, not from the sign of the
        row sum, or these valid rows get wrongly replaced with "unknown"."""
        pred_df = pd.DataFrame(
            {
                "emotion.acoustic_angry": [-2.0, 1.0],
                "emotion.acoustic_happy": [1.0, -1.0],
            }
        )
        mock_extractor = Mock()
        mock_extractor.extract.return_value = pred_df
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1, 2]})
        with use_context(_context("emotion.acoustic")):
            result = AudmodelPredictor(df).predict("train")

        assert list(result["emotion.acoustic_pred"]) == ["happy", "angry"]

    @patch("nkululeko.autopredict.ap_audmodel.FeatureExtractor")
    def test_overlapping_head_names_do_not_mix_columns(self, mock_feature_extractor):
        """Review on #468: with heads "foo" (categorical) and "foo_bar"
        (one-dimensional), naive startswith("foo_") matching for "foo"
        would also claim "foo_bar"'s own column as one of foo's own
        category scores, corrupting foo's argmax. Processing the longer
        head name first and removing its claimed column from the shared
        pool prevents that."""
        pred_df = pd.DataFrame(
            {
                "foo_angry": [0.1, 0.9],
                "foo_happy": [0.9, 0.1],
                "foo_bar": [5.0, 6.0],
            }
        )
        mock_extractor = Mock()
        mock_extractor.extract.return_value = pred_df
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1, 2]})
        with use_context(_context("['foo', 'foo_bar']")):
            result = AudmodelPredictor(df).predict("train")

        assert list(result["foo_pred"]) == ["happy", "angry"]
        assert list(result["foo_bar_pred"]) == [5.0, 6.0]

    @patch("nkululeko.autopredict.ap_audmodel.FeatureExtractor")
    def test_forces_no_reuse_to_avoid_stale_ad_hoc_cache(self, mock_feature_extractor):
        """Review on #468: AudmodelSet's whole-dataset cache is keyed only
        by database name + feats_type + model id/embeddings_name, never by
        the actual input rows -- the standalone `nkululeko.predict --model
        audmodel` CLI defaults every --list input to the same database
        name ("adhoc"), so without forcing a fresh extraction here, a
        second invocation on a different file list would silently reuse
        the first invocation's cached features."""
        pred_df = pd.DataFrame({"valence": [0.1]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = pred_df
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1]})
        context = _context("valence")
        with use_context(context):
            AudmodelPredictor(df).predict("train")

        assert context.config["FEATS"]["no_reuse"] == "True"

    @patch("nkululeko.autopredict.ap_audmodel.FeatureExtractor")
    def test_forces_no_reuse_even_without_an_existing_feats_section(
        self, mock_feature_extractor
    ):
        pred_df = pd.DataFrame({"valence": [0.1]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = pred_df
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1]})
        context = ExperimentContext(config={"DATA": {"databases": "['test_db']"}})
        with use_context(context):
            AudmodelPredictor(df).predict("train")

        assert context.config["FEATS"]["no_reuse"] == "True"
