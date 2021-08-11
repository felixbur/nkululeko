# xgrmodel.py

from xgboost.sklearn import XGBRegressor
from model import Model

class XGB_model(Model):
    """An XGBoost model"""
    clf = XGBRegressor() # set up the regressor

    def train(self):
        """Train the model"""
        self.clf.fit(self.feats_train.df, self.df_train['emotion'])

    def predict(self):
        """Predict the whole eval feature set"""
        return self.clf.predict(self.feats_test.df)