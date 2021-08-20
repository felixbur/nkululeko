# xgrmodel.py

from xgboost.sklearn import XGBRegressor
from model import Model
import glob_conf
from reporter import Reporter
from result import Result

class XGR_model(Model):
    """An XGBoost model"""
    clf = XGBRegressor() # set up the regressor

    def train(self):
        """Train the model"""
        target = glob_conf.config['DATA']['target']
        self.clf.fit(self.feats_train.df, self.df_train[target])

    def predict(self):
        """Predict the whole eval feature set"""
        predictions =  self.clf.predict(self.feats_test.df)
        report = Reporter(self.df_test[glob_conf.config['DATA']['target']].to_numpy().astype(float), predictions)
        report.result()
        return report
