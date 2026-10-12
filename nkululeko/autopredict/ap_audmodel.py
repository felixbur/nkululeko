"""
A generic predictor using an arbitrary audmodel (e.g. devAIce) output head.

GH #459 item 4: reuses the same FEATS.audmodel.id / FEATS.audmodel.embeddings_name
config keys the audmodel feature extractor itself reads (see
nkululeko.feat_extract.feats_audmodel.AudmodelSet), rather than inventing a
parallel [PREDICT]-section key.
"""

import ast

import numpy as np

from nkululeko.feat_extract.feats_audmodel import parse_embeddings_names
from nkululeko.feature_extractor import FeatureExtractor
from nkululeko.utils.util import Util


class AudmodelPredictor:
    """
    AudmodelPredictor
    writes one column per output dimension of each configured audmodel
    head, plus an argmax label column for any head with more than one
    dimension (treated as categorical, e.g. "emotion.acoustic").
    """

    def __init__(self, df):
        self.df = df
        self.util = Util("audmodelPredictor")

    def predict(self, split_selection):
        self.util.debug(f"predicting audmodel outputs for {split_selection} samples")
        feats_name = "_".join(ast.literal_eval(self.util.config["DATA"]["databases"]))
        self.feature_extractor = FeatureExtractor(
            self.df,
            ["audmodel"],
            feats_name,
            split_selection,
            context=self.util.context,
        )
        pred_df = self.feature_extractor.extract()
        return_df = self.df.copy()

        raw_embeddings_name = self.util.config_val(
            "FEATS", "audmodel.embeddings_name", "hidden_states"
        )
        for head in parse_embeddings_names(raw_embeddings_name):
            if head in pred_df.columns:
                return_df[f"{head}_pred"] = pred_df[head]
                continue
            head_cols = [c for c in pred_df.columns if c.startswith(f"{head}_")]
            if not head_cols:
                # Every row failed extraction for this head (extract()
                # returned no columns at all): no numeric value to report
                # for a continuous head, and no scores to argmax for a
                # categorical one.
                return_df[f"{head}_pred"] = "unknown"
                continue
            head_scores = pred_df[head_cols]
            # A row that failed extraction comes back all-NaN (see
            # feats_audmodel.py's AudmodelSet.extract()). A generic audmodel
            # head isn't guaranteed to output non-negative probabilities --
            # some models emit raw logits, where a valid row can legitimately
            # sum to <= 0 (e.g. [-2, 1]) -- so failure must be detected from
            # the NaNs themselves, not the sign of the sum (review on #468).
            # idxmax on an all-NaN row is also deprecated (pandas warns it
            # will raise ValueError in a future version), so fill with -inf
            # just for the argmax pick.
            all_failed = head_scores.isna().all(axis=1)
            pred_label = head_scores.fillna(-np.inf).idxmax(axis=1)
            pred_label = pred_label.where(~all_failed, "unknown")
            return_df[f"{head}_pred"] = pred_label.str.removeprefix(f"{head}_")

        return return_df
