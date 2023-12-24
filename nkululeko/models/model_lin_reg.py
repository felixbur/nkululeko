# model_knn_reg.py

from sklearn.linear_model import LinearRegression
from nkululeko.models.model import Model


class Lin_reg_model(Model):
    """An KNN model"""

    is_classifier = False

    def __init__(self, df_train, df_test, feats_train, feats_test):
        super().__init__(df_train, df_test, feats_train, feats_test)
        self.clf = LinearRegression()  # set up the classifier
