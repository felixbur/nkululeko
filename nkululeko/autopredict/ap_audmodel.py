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
    writes exactly one `<head>_pred` column per configured audmodel head:
    the raw value for a one-dimensional head (e.g. "valence"), or the
    argmax class label for a multi-dimension (categorical) head, e.g.
    "emotion.acoustic".
    """

    def __init__(self, df):
        self.df = df
        self.util = Util("audmodelPredictor")

    def predict(self, split_selection):
        self.util.debug(f"predicting audmodel outputs for {split_selection} samples")
        feats_name = "_".join(ast.literal_eval(self.util.config["DATA"]["databases"]))
        # AudmodelSet's whole-dataset cache is keyed only by database name +
        # feats_type + model id/embeddings_name, never by the actual input
        # rows (review on #468): fine for a real experiment (the same
        # DATA.databases across runs means the same data), but the
        # standalone `nkululeko.predict --model audmodel` CLI defaults
        # every --list input to the same database name ("adhoc"), so a
        # second invocation on a different file list would otherwise
        # silently reuse the first invocation's cached features. Force a
        # fresh extraction every time for this ad-hoc path.
        if "FEATS" not in self.util.config:
            self.util.config["FEATS"] = {}
        self.util.config["FEATS"]["no_reuse"] = "True"
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
        heads = parse_embeddings_names(raw_embeddings_name)
        # Longest head name first, claiming each matched column out of a
        # shared pool as we go (review on #468): a shorter head's name can
        # be a strict prefix of a longer one (e.g. "foo" and "foo_bar"), so
        # matching heads in arbitrary/configured order would let "foo"'s
        # startswith("foo_") also claim "foo_bar"'s own column, corrupting
        # "foo"'s argmax with a score that belongs to a different head.
        # Processing the longest names first and removing claimed columns
        # from the pool means a later, shorter head can never re-claim a
        # column a longer head already took.
        remaining_columns = list(pred_df.columns)
        for head in sorted(heads, key=len, reverse=True):
            if head in remaining_columns:
                return_df[f"{head}_pred"] = pred_df[head]
                remaining_columns.remove(head)
                continue
            head_cols = [c for c in remaining_columns if c.startswith(f"{head}_")]
            for c in head_cols:
                remaining_columns.remove(c)
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
