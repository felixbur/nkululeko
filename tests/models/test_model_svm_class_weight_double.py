"""Regression for GH #448: with [MODEL] class_weight = True, SVM_model's
constructor passed class_weight="balanced" into svm.SVC(...) *and*
Model.train() separately fit with a balanced sample_weight. sklearn
multiplies class_weight by sample_weight internally, so each class ended
up weighted by the square of its balanced weight instead of the balanced
weight itself, pushing decisions far more aggressively toward the
minority class than [MODEL] class_weight = True is supposed to.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.utils.class_weight import compute_sample_weight

from nkululeko.experiment_context import ExperimentContext, use_context
from nkululeko.models.model_svm import SVM_model


def _make_imbalanced_data():
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
    df = pd.DataFrame({"emotion": [str(v) for v in y]})
    feats = pd.DataFrame(X, index=df.index)
    return df, feats


class TestSvmConstructorDoesNotSetClassWeight:
    def test_class_weight_not_set_on_clf_even_when_requested(self):
        context = ExperimentContext(
            config={"MODEL": {"class_weight": "True", "C_val": "0.1"}}
        )
        df, feats = _make_imbalanced_data()
        with use_context(context):
            model = SVM_model(df, df, feats, feats)
        assert model.clf.get_params()["class_weight"] is None


class TestSvmTrainMatchesSingleWeightedSklearn:
    def test_trained_svm_matches_sample_weight_only_reference(self):
        context = ExperimentContext(
            config={"MODEL": {"class_weight": "True", "C_val": "0.1"}}
        )
        df, feats = _make_imbalanced_data()
        with use_context(context):
            model = SVM_model(df, df, feats, feats)
            model.df_train = df
            model.feats_train = feats
            model.train()

        # Reference: sklearn SVC weighted exactly once, via sample_weight
        # only (matches the intent of [MODEL] class_weight = True).
        reference = svm.SVC(
            kernel="rbf", C=0.1, gamma="scale", probability=True, random_state=42
        )
        sample_weight = compute_sample_weight(class_weight="balanced", y=df["emotion"])
        reference.fit(feats.to_numpy(), df["emotion"], sample_weight=sample_weight)

        # Double-weighted reference reproducing the pre-fix bug: balanced
        # class_weight in the constructor *and* balanced sample_weight.
        double_weighted = svm.SVC(
            kernel="rbf",
            C=0.1,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )
        double_weighted.fit(feats.to_numpy(), df["emotion"], sample_weight=sample_weight)

        trained_preds = model.clf.predict(feats.to_numpy())
        reference_preds = reference.predict(feats.to_numpy())
        double_weighted_preds = double_weighted.predict(feats.to_numpy())

        assert np.array_equal(trained_preds, reference_preds)
        # Sanity check the two references actually diverge on this data,
        # i.e. the bug this guards against is real and reproducible here.
        assert not np.array_equal(reference_preds, double_weighted_preds)
