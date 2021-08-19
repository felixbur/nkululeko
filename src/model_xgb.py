# xgbmodel.py

from xgboost import XGBClassifier
from model import Model
import glob_conf

class XGB_model(Model):
    """An XGBoost model"""
    clf = XGBClassifier() # set up the classifier

    def train(self):
        """Train the model"""
        target = glob_conf.config['DATA']['target']
        self.clf.fit(self.feats_train.df.to_numpy(), self.df_train[target])

    def predict(self):
        """Predict the whole eval feature set"""
        return self.clf.predict(self.feats_test.df.to_numpy())