# svrmodel.py

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVR
from model import Model
import glob_conf
from reporter import Reporter
from sklearn.model_selection import GridSearchCV
import ast

class SVR_model(Model):
    """An SVR model"""
    def __init__(self, df_train, df_test, feats_train, feats_test):
        super().__init__(df_train, df_test, feats_train, feats_test)
        self.clf = SVR(kernel='rbf') # set up the classifier

    def set_C(self, c):
        """Set the C parameter"""
        self.clf.C = c
