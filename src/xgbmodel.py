# xgbmodel.py

from xgboost import XGBClassifier
from model import Model

class XGB_model(Model):
    """An XGBoost model"""
    clf = XGBClassifier() # set up the classifier

    def train(self):
        """Train the model"""
        target = self.config['DATA']['target']
        self.clf.fit(self.feats_train.df, self.df_train[target])

    def predict(self):
        """Predict the whole eval feature set"""
        return self.clf.predict(self.feats_test.df)