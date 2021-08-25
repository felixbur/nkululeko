# xgbmodel.py

from xgboost import XGBClassifier
from model import Model
import glob_conf
from reporter import Reporter
from result import Result
from sklearn.model_selection import GridSearchCV
import ast

class XGB_model(Model):
    """An XGBoost model"""
    clf = XGBClassifier() # set up the classifier

    def train(self):
        """Train the model"""
        target = glob_conf.config['DATA']['target']
        if self.feats_train.df.isna().to_numpy().any():
            self.feats_train.df.to_pickle('feats_train.df')
            self.util.error('NANs exist')
        feats = self.feats_train.df.to_numpy()
        try:
            # tune the model meta parameters
            tuning_params = ast.literal_eval(glob_conf.config['MODEL']['tuning_params'])
            tuned_params={}
            scoring = glob_conf.config['MODEL']['scoring']
            for param in tuning_params:
                values = ast.literal_eval(glob_conf.config['MODEL'][param])
                tuned_params[param] = values
            self.util.debug(f'tuning on {tuned_params}')
            self.clf = GridSearchCV(self.clf, tuned_params, refit = True, verbose = 3, scoring=scoring)
            self.clf.fit(feats, self.df_train[target], sample_weight=self.classes_weights)
            self.util.debug(f'winner parameters: {self.clf.best_params_}')
        except KeyError:
            self.clf.fit(feats, self.df_train[target], sample_weight=self.classes_weights)

    def predict(self):
        """Predict the whole eval feature set"""
        predictions = self.clf.predict(self.feats_test.df.to_numpy())
        report = Reporter(self.df_test[glob_conf.config['DATA']['target']], predictions)
        return report

