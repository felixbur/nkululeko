# xgrmodel.py

from xgboost.sklearn import XGBRegressor

from nkululeko.models.model import Model


class XGR_model(Model):
    """An XGBoost regression model"""

    def __init__(self, df_train, df_test, feats_train, feats_test):
        super().__init__(df_train, df_test, feats_train, feats_test)
        self.name = "xgr"
        self.is_classifier = False
        self.clf = XGBRegressor()  # set up the regressor
