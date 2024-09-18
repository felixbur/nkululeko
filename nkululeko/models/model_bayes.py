# model_bayes.py

from sklearn.naive_bayes import GaussianNB

from nkululeko.models.model import Model


class Bayes_model(Model):
    is_classifier = True

    """A Bayesian model"""

    def __init__(self, df_train, df_test, feats_train, feats_test):
        super().__init__(df_train, df_test, feats_train, feats_test)
        self.clf = GaussianNB()  # set up the classifier
        self.name = "bayes"
