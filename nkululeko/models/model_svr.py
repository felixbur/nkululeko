# svrmodel.py

from sklearn import svm

from nkululeko.models.model import Model


class SVR_model(Model):
    """An SVR model"""

    is_classifier = False

    def __init__(self, df_train, df_test, feats_train, feats_test):
        super().__init__(df_train, df_test, feats_train, feats_test)
        self.name = "svr"
        c = float(self.util.config_val("MODEL", "C_val", "0.001"))
        #  kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’
        kernel = self.util.config_val("MODEL", "kernel", "rbf")
        self.clf = svm.SVR(kernel=kernel, C=c)  # set up the classifier

    def set_C(self, c):
        """Set the C parameter"""
        self.clf.C = c
