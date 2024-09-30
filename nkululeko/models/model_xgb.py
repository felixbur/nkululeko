# xgbmodel.py

from xgboost import XGBClassifier

from nkululeko.models.model import Model


class XGB_model(Model):
    """An XGBoost model"""

    def __init__(self, df_train, df_test, feats_train, feats_test):
        super().__init__(df_train, df_test, feats_train, feats_test)
        self.name = "xgb"
        self.is_classifier = True
        self.clf = XGBClassifier()  # set up the classifier

    def get_type(self):
        return "xgb"
