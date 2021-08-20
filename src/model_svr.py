# svrmodel.py

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVR
from model import Model
import glob_conf
from reporter import Reporter
from result import Result

class SVR_model(Model):
    """An SVR model"""
    def __init__(self, df_train, df_test, feats_train, feats_test):
        super().__init__(df_train, df_test, feats_train, feats_test)
        c = float(self.util.config_val('MODEL', 'C', '0.001'))
        self.clf = SVR(kernel='rbf', C=c) # set up the classifier

    def set_C(self, c):
        """Set the C parameter"""
        self.clf.C = c

    def train(self):
        """Train the model"""
        target = glob_conf.config['DATA']['target']
        if self.feats_train.df.isna().to_numpy().any():
            self.util.error('NAN')
        feats = self.feats_train.df.to_numpy()
        self.clf.fit(feats, self.df_train[target])

    def predict(self):
        """Predict the whole eval feature set"""
        predictions = self.clf.predict(self.feats_test.df.to_numpy())
        report = Reporter(self.df_test[glob_conf.config['DATA']['target']], predictions)
        report.result()
        return report
