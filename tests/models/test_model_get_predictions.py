"""Regression for GH #440: Model.get_predictions() (nkululeko/models/model.py)
took the predicted label as argmax(predict_proba(...)) rather than
predict(...). For SVC(probability=True), predict_proba comes from a
separate Platt-scaling calibration fit that can re-learn its own class
prior independently of the decision function -- so with
[MODEL] class_weight = True on imbalanced data, class_weight correctly
shifts the SVM decision function (predict()), but argmax over the
calibrated probabilities threw that shift away and fell back to
predicting close to the majority class regardless.

This reproduces the divergence with a real, fitted SVC (not a mock), on
synthetic imbalanced data chosen to reliably reproduce the exact reported
symptom: argmax(predict_proba) collapses back toward the training prior,
while predict() (which reads the class_weight-shifted decision function)
does not.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

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


@pytest.fixture
def fitted_balanced_svm():
    context = ExperimentContext(
        config={"MODEL": {"class_weight": "True", "C_val": "0.1"}}
    )
    df, feats = _make_imbalanced_data()
    with use_context(context):
        model = SVM_model(df, df, feats, feats)
        model.clf.fit(feats.to_numpy(), df["emotion"])
        yield model, feats


class TestGetPredictionsUsesPredictNotArgmaxProba:
    def test_predict_proba_argmax_would_disagree_with_predict(
        self, fitted_balanced_svm
    ):
        # Sanity check that this scenario actually reproduces the
        # underlying sklearn divergence this test guards against - if this
        # ever stops reproducing (e.g. sklearn changes SVC internals), the
        # test below would pass vacuously without actually exercising the
        # fix.
        model, feats = fitted_balanced_svm
        native = model.clf.predict(feats.to_numpy())
        proba = model.clf.predict_proba(feats.to_numpy())
        argmax_proba = model.clf.classes_[np.argmax(proba, axis=1)]
        assert (native != argmax_proba).sum() > 0

    def test_get_predictions_matches_native_predict(self, fitted_balanced_svm):
        model, feats = fitted_balanced_svm
        predictions, probas = model.get_predictions()

        native = model.clf.predict(feats.to_numpy())
        np.testing.assert_array_equal(predictions, native)

        # predict_proba is still the right source for the probability
        # columns themselves - just not for which label "wins".
        assert probas is not None
        assert set(probas.columns) == set(model.clf.classes_)

    def test_get_predictions_no_longer_matches_naive_argmax_proba(
        self, fitted_balanced_svm
    ):
        model, feats = fitted_balanced_svm
        predictions, _ = model.get_predictions()

        proba = model.clf.predict_proba(feats.to_numpy())
        argmax_proba = model.clf.classes_[np.argmax(proba, axis=1)]
        # This is the actual bug: predictions used to equal this exactly.
        assert (predictions != argmax_proba).sum() > 0

    def test_class_weight_recovers_reasonable_sensitivity(self, fitted_balanced_svm):
        # End-to-end sanity check matching the issue's reported impact:
        # class_weight="balanced" should noticeably raise the positive
        # rate compared to the (collapsed-to-prior) argmax(predict_proba)
        # behavior, since predict() reflects the shifted decision function.
        model, feats = fitted_balanced_svm
        predictions, _ = model.get_predictions()
        positive_rate = (predictions == "1").mean()

        proba = model.clf.predict_proba(feats.to_numpy())
        argmax_proba = model.clf.classes_[np.argmax(proba, axis=1)]
        argmax_positive_rate = (argmax_proba == "1").mean()

        assert positive_rate > argmax_positive_rate
