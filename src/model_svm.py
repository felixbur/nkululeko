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
        c = float(self.util.config_val('MODEL', 'C_val', '0.001'))
        self.clf = svm.SVC(kernel='linear', C=c, gamma='auto') # set up the classifier

    def set_C(self, c):
        """Set the C parameter"""
        self.clf.C = c
