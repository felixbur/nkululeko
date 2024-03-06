# model_svm.py

from sklearn import svm
from nkululeko.models.model import Model


class SVM_model(Model):
    """An SVM model"""

    is_classifier = True

    def __init__(self, df_train, df_test, feats_train, feats_test):
        super().__init__(df_train, df_test, feats_train, feats_test)
        c = float(self.util.config_val("MODEL", "C_val", "0.001"))
        self.clf = svm.SVC(
            kernel="linear", C=c, gamma="scale", probability=True
        )  # set up the classifier

    def set_C(self, c):
        """Set the C parameter"""
        self.clf.C = c
