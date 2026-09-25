"""Regression for GH #448 review feedback: Model.train() returns early into
_do_logo()/_x_fold_cross() for [MODEL] logo / k_fold_cross, both of which
called self.clf.fit(...) with no sample_weight at all. Before #448's fix,
SVM_model baked class_weight="balanced" into the SVC constructor itself, so
these paths got balanced weighting "for free" through the estimator's own
stored parameter. Removing that constructor fallback (to stop double-
weighting the generic sample_weight path) silently dropped weighting
entirely for LOGO/k-fold-cross SVM runs -- this guards that both helpers now
compute and pass a balanced sample_weight per fold themselves.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from nkululeko.experiment_context import ExperimentContext, use_context
from nkululeko.models.model_svm import SVM_model


def _make_imbalanced_data(n_speakers=30):
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.85, 0.15],
        flip_y=0.05,
        random_state=7,
    )
    df = pd.DataFrame({"emotion": y.astype(int)})
    df["speaker"] = [f"spk{i % n_speakers}" for i in range(len(df))]
    feats = pd.DataFrame(X, index=df.index)
    half = len(df) // 2
    return (
        df.iloc[:half],
        df.iloc[half:],
        feats.iloc[:half],
        feats.iloc[half:],
    )


def _spy_on_fit(model):
    calls = []
    original_fit = model.clf.fit

    def spy_fit(X, y, sample_weight=None):
        calls.append(sample_weight)
        return original_fit(X, y, sample_weight=sample_weight)

    model.clf.fit = spy_fit
    return calls


class TestKFoldCrossPassesSampleWeight:
    def test_class_weight_true_passes_balanced_sample_weight_per_fold(self):
        context = ExperimentContext(
            config={
                "MODEL": {
                    "class_weight": "True",
                    "C_val": "0.1",
                    "k_fold_cross": "3",
                }
            }
        )
        df_train, df_test, feats_train, feats_test = _make_imbalanced_data()
        with use_context(context):
            model = SVM_model(df_train, df_test, feats_train, feats_test)
            calls = _spy_on_fit(model)
            model.train()

        assert len(calls) == 3
        for sample_weight in calls:
            assert sample_weight is not None
            assert len(np.unique(sample_weight)) > 1

    def test_class_weight_false_passes_no_sample_weight(self):
        context = ExperimentContext(
            config={"MODEL": {"C_val": "0.1", "k_fold_cross": "3"}}
        )
        df_train, df_test, feats_train, feats_test = _make_imbalanced_data()
        with use_context(context):
            model = SVM_model(df_train, df_test, feats_train, feats_test)
            calls = _spy_on_fit(model)
            model.train()

        assert len(calls) == 3
        assert all(sample_weight is None for sample_weight in calls)


class TestLogoPassesSampleWeight:
    def test_class_weight_true_passes_balanced_sample_weight_per_group(
        self, tmp_path
    ):
        context = ExperimentContext(
            config={
                "MODEL": {
                    "class_weight": "True",
                    "C_val": "0.1",
                    "logo": "3",
                },
                "EXP": {"root": str(tmp_path), "name": "test_logo"},
            }
        )
        df_train, df_test, feats_train, feats_test = _make_imbalanced_data()
        with use_context(context):
            model = SVM_model(df_train, df_test, feats_train, feats_test)
            calls = _spy_on_fit(model)
            model.train()

        assert len(calls) == 3
        for sample_weight in calls:
            assert sample_weight is not None
            assert len(np.unique(sample_weight)) > 1
