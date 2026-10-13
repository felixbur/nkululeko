from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from nkululeko.autopredict.ap_audmodel import AudmodelPredictor
from nkululeko.experiment_context import ExperimentContext, use_context


def _context():
    return ExperimentContext(config={"DATA": {"databases": "['test_db']"}})


def _mock_extractor(pred_df, embeddings_names, head_label_counts=None):
    """Stand-in for FeatureExtractor: .extract() returns pred_df, and
    .feat_extractor is the underlying AudmodelSet-shaped object
    AudmodelPredictor reads the head-to-columns boundary from (review on
    #468: this boundary must come from the extractor itself, not be
    reconstructed from column-name string matching)."""
    mock_extractor = Mock()
    mock_extractor.extract.return_value = pred_df
    mock_extractor.feat_extractor.embeddings_names = embeddings_names
    if head_label_counts is not None:
        mock_extractor.feat_extractor._head_label_counts = head_label_counts
    return mock_extractor


class TestAudmodelPredictor:
    def test_init(self):
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = AudmodelPredictor(df)

        assert predictor.df is df
        assert predictor.util is not None

    @patch("nkululeko.autopredict.ap_audmodel.FeatureExtractor")
    def test_single_dim_head_is_copied_as_is(self, mock_feature_extractor):
        pred_df = pd.DataFrame({"valence": [0.1, 0.5, 0.9]})
        mock_feature_extractor.return_value = _mock_extractor(pred_df, ["valence"])

        df = pd.DataFrame({"dummy": [1, 2, 3]})
        with use_context(_context()):
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
        mock_feature_extractor.return_value = _mock_extractor(
            pred_df, ["emotion.acoustic"]
        )

        df = pd.DataFrame({"dummy": [1, 2, 3]})
        with use_context(_context()):
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
        mock_feature_extractor.return_value = _mock_extractor(
            pred_df, ["valence", "emotion.acoustic"], head_label_counts=[1, 2]
        )

        df = pd.DataFrame({"dummy": [1, 2]})
        with use_context(_context()):
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
        mock_feature_extractor.return_value = _mock_extractor(pred_df, ["valence"])

        df = pd.DataFrame({"dummy": [1, 2, 3]})
        with use_context(_context()):
            result = AudmodelPredictor(df).predict("train")

        assert list(result["valence_pred"]) == ["unknown", "unknown", "unknown"]

    @patch("nkululeko.autopredict.ap_audmodel.FeatureExtractor")
    def test_one_of_several_heads_missing_entirely_falls_back_to_unknown(
        self, mock_feature_extractor
    ):
        """Same as above, but for one head out of several (the dataframe
        is short just that head's columns, not empty altogether)."""
        pred_df = pd.DataFrame({"valence": [0.1, 0.9]})
        mock_feature_extractor.return_value = _mock_extractor(
            pred_df, ["valence", "emotion.acoustic"], head_label_counts=[1, 2]
        )

        df = pd.DataFrame({"dummy": [1, 2]})
        with use_context(_context()):
            result = AudmodelPredictor(df).predict("train")

        assert list(result["valence_pred"]) == [0.1, 0.9]
        assert list(result["emotion.acoustic_pred"]) == ["unknown", "unknown"]

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
        mock_feature_extractor.return_value = _mock_extractor(
            pred_df, ["emotion.acoustic"]
        )

        df = pd.DataFrame({"dummy": [1, 2]})
        with use_context(_context()):
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
        mock_feature_extractor.return_value = _mock_extractor(
            pred_df, ["emotion.acoustic"]
        )

        df = pd.DataFrame({"dummy": [1, 2]})
        with use_context(_context()):
            result = AudmodelPredictor(df).predict("train")

        assert list(result["emotion.acoustic_pred"]) == ["happy", "angry"]

    @patch("nkululeko.autopredict.ap_audmodel.FeatureExtractor")
    def test_overlapping_head_names_do_not_mix_columns(self, mock_feature_extractor):
        """Review on #468: with heads "foo" (categorical) and "foo_bar"
        (one-dimensional), column-name string matching (however the heads
        are ordered) can misattribute columns between them. Reading the
        boundary directly from the extractor's own per-head column
        counts -- not reconstructing it from the names -- sidesteps that
        entirely."""
        pred_df = pd.DataFrame(
            {
                "foo_angry": [0.1, 0.9],
                "foo_happy": [0.9, 0.1],
                "foo_bar": [5.0, 6.0],
            }
        )
        mock_feature_extractor.return_value = _mock_extractor(
            pred_df, ["foo", "foo_bar"], head_label_counts=[2, 1]
        )

        df = pd.DataFrame({"dummy": [1, 2]})
        with use_context(_context()):
            result = AudmodelPredictor(df).predict("train")

        assert list(result["foo_pred"]) == ["happy", "angry"]
        assert list(result["foo_bar_pred"]) == [5.0, 6.0]

    @patch("nkululeko.autopredict.ap_audmodel.FeatureExtractor")
    def test_overlapping_head_names_with_ambiguous_label_do_not_mix_columns(
        self, mock_feature_extractor
    ):
        """Review on #468: this is the case that broke prefix matching
        even with longest-name-first ordering -- head "foo" has a label
        literally named "bar_angry", producing a column "foo_bar_angry"
        that looks exactly like it belongs to head "foo_bar". Positional
        slicing by the extractor's own recorded per-head counts isn't
        fooled by this, since it never inspects the column text at all."""
        pred_df = pd.DataFrame(
            {
                "foo_bar_angry": [0.9, 0.1],  # head "foo"'s own column
                "foo_bar_x": [0.2, 0.8],
                "foo_bar_y": [0.8, 0.2],
            }
        )
        mock_feature_extractor.return_value = _mock_extractor(
            pred_df, ["foo", "foo_bar"], head_label_counts=[1, 2]
        )

        df = pd.DataFrame({"dummy": [1, 2]})
        with use_context(_context()):
            result = AudmodelPredictor(df).predict("train")

        assert list(result["foo_pred"]) == [0.9, 0.1]
        assert list(result["foo_bar_pred"]) == ["y", "x"]

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
        mock_feature_extractor.return_value = _mock_extractor(pred_df, ["valence"])

        df = pd.DataFrame({"dummy": [1]})
        context = _context()
        with use_context(context):
            AudmodelPredictor(df).predict("train")

        assert context.config["FEATS"]["no_reuse"] == "True"

    @patch("nkululeko.autopredict.ap_audmodel.FeatureExtractor")
    def test_forces_no_reuse_even_without_an_existing_feats_section(
        self, mock_feature_extractor
    ):
        pred_df = pd.DataFrame({"valence": [0.1]})
        mock_feature_extractor.return_value = _mock_extractor(pred_df, ["valence"])

        df = pd.DataFrame({"dummy": [1]})
        context = ExperimentContext(config={"DATA": {"databases": "['test_db']"}})
        with use_context(context):
            AudmodelPredictor(df).predict("train")

        assert context.config["FEATS"]["no_reuse"] == "True"

    @patch("nkululeko.autopredict.ap_audmodel.FeatureExtractor")
    def test_disables_feats_features_during_extraction_and_restores_it(
        self, mock_feature_extractor
    ):
        """Review on #468: FEATS.features (a generic feature-subset/reorder
        filter, applied inside Featureset.filter()) would reshuffle/drop
        audmodel's own output columns before this predictor ever sees
        them, breaking the positional head->column mapping. It must be
        disabled just for this extraction and restored afterward, since
        something else in the same run might still need it."""
        pred_df = pd.DataFrame({"valence": [0.1]})
        mock_extractor = _mock_extractor(pred_df, ["valence"])
        captured = {}
        context = _context()
        context.config["FEATS"] = {"features": "['valence']"}

        def fake_extract():
            captured["features_during_extract"] = "features" in context.config["FEATS"]
            return pred_df

        mock_extractor.extract.side_effect = fake_extract
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1]})
        with use_context(context):
            AudmodelPredictor(df).predict("train")

        assert captured["features_during_extract"] is False
        assert context.config["FEATS"]["features"] == "['valence']"

    @patch("nkululeko.autopredict.ap_audmodel.FeatureExtractor")
    def test_no_feats_features_key_is_a_no_op(self, mock_feature_extractor):
        pred_df = pd.DataFrame({"valence": [0.1]})
        mock_feature_extractor.return_value = _mock_extractor(pred_df, ["valence"])

        df = pd.DataFrame({"dummy": [1]})
        context = _context()
        with use_context(context):
            AudmodelPredictor(df).predict("train")

        assert "features" not in context.config["FEATS"]
