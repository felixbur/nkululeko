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

    Every multi-dimension head is assumed categorical (its columns are
    class scores) -- there is no reliable way to tell a categorical head
    apart from a continuous multi-dimensional one (e.g. a raw embedding)
    from audonnx's own metadata alone (review on #468). This matches the
    devAIce-style heads #459 was built around (arousal/valence/
    emotion.acoustic/...), each a single dimension or a small finite set
    of classes. Configuring FEATS.audmodel.embeddings_name to a
    continuous, non-categorical head (notably the "hidden_states" default,
    meant for plain feature extraction, not this predictor) produces a
    meaningless argmax "prediction" -- pick real class/dimension heads
    when using this PREDICT target.
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
        # standalone `nkululeko.predict --model audmodel_pred` CLI defaults
        # every --list input to the same database name ("adhoc"), so a
        # second invocation on a different file list would otherwise
        # silently reuse the first invocation's cached features. Force a
        # fresh extraction every time for this ad-hoc path -- this also
        # guarantees AudmodelSet._load_model() actually runs below (not
        # skipped by a cache hit), which self._head_label_counts relies on.
        if "FEATS" not in self.util.config:
            self.util.config["FEATS"] = {}
        feats_section = self.util.config["FEATS"]
        feats_section["no_reuse"] = "True"
        self.feature_extractor = FeatureExtractor(
            self.df,
            ["audmodel"],
            feats_name,
            split_selection,
            context=self.util.context,
        )
        # FEATS.features (a generic feature-subset/reorder filter applied
        # by Featureset.filter(), e.g. to restrict a model to a hand-picked
        # subset) runs inside extract() below and would reshuffle/drop
        # audmodel's own output columns before this predictor ever sees
        # them, breaking the positional head->column mapping that follows
        # (review on #468). This predictor's whole job is to turn
        # audmodel's complete output into <head>_pred columns, so disable
        # that filter for the duration of this extraction, restoring it
        # afterward in case something else in the same run still needs it.
        had_features_key = "features" in feats_section
        original_features = feats_section.get("features") if had_features_key else None
        if had_features_key:
            del feats_section["features"]
        try:
            pred_df = self.feature_extractor.extract()
        finally:
            if had_features_key:
                feats_section["features"] = original_features
        return_df = self.df.copy()

        # Read the head-to-columns boundary straight from the AudmodelSet
        # instance that did the extraction, instead of reconstructing
        # ownership from column-name string matching (review on #468):
        # with two categorical heads whose names overlap as a string
        # prefix (e.g. "foo" and "foo_bar"), a label on the shorter head
        # can itself look like "bar_angry", producing a column
        # "foo_bar_angry" that any prefix-based heuristic -- however the
        # heads are ordered -- would wrongly attribute to "foo_bar". The
        # extractor already knows the true, unambiguous boundary (each
        # head's own column count, recorded at load time in the same
        # order as its own embeddings_names).
        audmodel_set = self.feature_extractor.feat_extractor
        heads = getattr(audmodel_set, "embeddings_names", None)
        if heads is None:
            # _load_model() is lazy (feats_audmodel.py, review on #468):
            # it only runs on the first file that actually reaches
            # extraction, so if literally every file failed before that
            # point (e.g. every audio file was unreadable), it never ran
            # at all and embeddings_names/_head_label_counts were never
            # set. Parse the configured head names directly so this still
            # reports "unknown" per head instead of crashing.
            raw_embeddings_name = self.util.config_val(
                "FEATS", "audmodel.embeddings_name", "hidden_states"
            )
            heads = parse_embeddings_names(raw_embeddings_name)
            for head in heads:
                return_df[f"{head}_pred"] = "unknown"
            return return_df
        if len(heads) == 1:
            # Single head: every column belongs to it, whether ONNX or
            # torch-backed -- _head_label_counts is only ever set on the
            # ONNX path (multi-head is rejected outright for torch, see
            # AudmodelSet._load_model()), so this also covers single-head
            # torch models, which have no such bookkeeping.
            self._write_head_pred(return_df, heads[0], pred_df, list(pred_df.columns))
            return return_df

        pos = 0
        for head, count in zip(heads, audmodel_set._head_label_counts):
            if pos + count > len(pred_df.columns):
                head_cols = []
            else:
                head_cols = list(pred_df.columns[pos : pos + count])
                pos += count
            self._write_head_pred(return_df, head, pred_df, head_cols)

        return return_df

    def _write_head_pred(self, return_df, head, pred_df, head_cols):
        if not head_cols:
            # Every row failed extraction for this head (extract()
            # returned no columns at all, or fewer than expected): no
            # numeric value to report for a continuous head, and no
            # scores to argmax for a categorical one.
            return_df[f"{head}_pred"] = "unknown"
            return
        if len(head_cols) == 1:
            return_df[f"{head}_pred"] = pred_df[head_cols[0]]
            return
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
