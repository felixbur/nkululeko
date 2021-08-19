# svmmodel.py

import numpy as np
import pandas as pd
from sklearn import svm
from model import Model

class SVM_model(Model):
    """An SVM model"""
    def __init__(self, config, df_train, df_test, feats_train, feats_test):
        super().__init__(config, df_train, df_test, feats_train, feats_test)
        c = float(self.util.config_val('MODEL', 'C', '0.001'))
        self.clf = svm.SVC(kernel='linear', C=c) # set up the classifier

    def set_C(self, c):
        """Set the C parameter"""
        self.clf.C = c

    def train(self):
        """Train the model"""
        target = self.config['DATA']['target']
        if self.feats_train.df.isna().to_numpy().any():
            self.util.error('NAN')
        feats = self.feats_train.df.to_numpy()
        self.clf.fit(feats, self.df_train[target])

    def predict(self):
        """Predict the whole eval feature set"""
        return self.clf.predict(self.feats_test.df.to_numpy())