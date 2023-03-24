# model_bayes.py

from sklearn.naive_bayes import GaussianNB
from nkululeko.model import Model

class Bayes_model(Model):
    is_classifier = True

    """An SVM model"""
    def __init__(self, df_train, df_test, feats_train, feats_test):
        super().__init__(df_train, df_test, feats_train, feats_test)
        c = float(self.util.config_val('MODEL', 'C_val', '0.001'))
        self.clf = GaussianNB() # set up the classifier

