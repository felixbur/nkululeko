import os

import numpy as np
import pandas as pd
import pytest

from nkululeko.models.model_svm import SVM_model


@pytest.fixture(scope="module")
def polish_data():
    data_dir = os.path.abspath("data/polish")
    train_csv = os.path.join(data_dir, "polish_train.csv")
    test_csv = os.path.join(data_dir, "polish_test.csv")
    # Load a small sample for speed
    df_train = pd.read_csv(train_csv).head(10)
    df_test = pd.read_csv(test_csv).head(5)
    # Assume 'file' and 'emotion' columns exist
    # Mock features: random floats, shape (n_samples, n_features)
    feats_train = np.random.rand(len(df_train), 10)
    feats_test = np.random.rand(len(df_test), 10)
    return df_train, df_test, feats_train, feats_test


def test_svm_model_init(polish_data):
    df_train, df_test, feats_train, feats_test = polish_data
    model = SVM_model(df_train, df_test, feats_train, feats_test)
    assert model.name == "svm"
    assert hasattr(model, "clf")
    assert model.is_classifier


def test_svm_model_fit_and_predict(polish_data):
    df_train, df_test, feats_train, feats_test = polish_data
    model = SVM_model(df_train, df_test, feats_train, feats_test)
    # Fit the model
    y_train = df_train["emotion"].astype(str)
    model.clf.fit(feats_train, y_train)
    # Predict
    preds = model.clf.predict(feats_test)
    assert len(preds) == feats_test.shape[0]


def test_svm_model_set_c(polish_data):
    df_train, df_test, feats_train, feats_test = polish_data
    model = SVM_model(df_train, df_test, feats_train, feats_test)
    old_c = model.clf.C
    model.set_c(2.0)
    assert model.clf.C == 2.0
    assert model.clf.C != old_c


def test_svm_model_get_type(polish_data):
    df_train, df_test, feats_train, feats_test = polish_data
    model = SVM_model(df_train, df_test, feats_train, feats_test)
    assert model.get_type() == "svm"
