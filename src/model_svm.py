# svmmodel.py

import numpy as np
import pandas as pd
from sklearn import svm
from model import Model
import glob_conf
from reporter import Reporter
from sklearn.model_selection import GridSearchCV
import ast

class SVM_model(Model):
    """An SVM model"""
    def __init__(self, df_train, df_test, feats_train, feats_test):
        super().__init__(df_train, df_test, feats_train, feats_test)
        self.clf = svm.SVC(kernel='linear') # set up the classifier

    def set_C(self, c):
        """Set the C parameter"""
        self.clf.C = c

    def train(self):
        """Train the model"""
        target = glob_conf.config['DATA']['target']
        if self.feats_train.df.isna().to_numpy().any():
            self.util.error('NANs exist')
        feats = self.feats_train.df.to_numpy()
        try:
            # tune the model meta parameters
            tuning_params = ast.literal_eval(glob_conf.config['MODEL']['tuning_params'])
            tuned_params={}
            scoring = glob_conf.config['MODEL']['scoring']
            for param in tuning_params:
                values = ast.literal_eval(glob_conf.config['MODEL'][param])
                tuned_params[param] = values
            self.util.debug(f'tuning on {tuned_params}')
            self.clf = GridSearchCV(self.clf, tuned_params, refit = True, verbose = 3, scoring=scoring)
            self.clf.fit(feats, self.df_train[target])
            self.util.debug(f'winner parameters: {self.clf.best_params_}')        
        except KeyError:
            c = float(self.util.config_val('MODEL', 'C', '0.001'))
            self.set_C(c)
            self.clf.fit(feats, self.df_train[target])
        

    def predict(self):
        """Predict the whole eval feature set"""
        predictions = self.clf.predict(self.feats_test.df.to_numpy())
        report = Reporter(self.df_test[glob_conf.config['DATA']['target']], predictions)
        return report