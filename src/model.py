# model.py
from util import Util 
import glob_conf
from sklearn.utils import class_weight
from reporter import Reporter
import ast
from sklearn.model_selection import GridSearchCV


class Model:
    """Generic model class"""

    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor taking the configuration and all dataframes"""
        self.df_train, self.df_test, self.feats_train, self.feats_test = df_train, df_test, feats_train, feats_test
        self.util = Util()
        target = glob_conf.config['DATA']['target']
        self.classes_weights = class_weight.compute_sample_weight(
            class_weight='balanced',
            y=df_train[target]
        )
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
            try: 
                class_weight = self.util.config_val('MODEL', 'class_weight', 0)
                if class_weight:
                    self.util.debug('using class weight')
                    self.clf.fit(feats, self.df_train[target], sample_weight=self.classes_weights)
            except KeyError:
                self.clf.fit(feats, self.df_train[target])
            self.util.debug(f'winner parameters: {self.clf.best_params_}')
        except KeyError:
            try: 
                class_weight = glob_conf.config['MODEL']['class_weight']
                if class_weight:
                    self.util.debug('using class weight')
                    self.clf.fit(feats, self.df_train[target], sample_weight=self.classes_weights)
            except KeyError:
                self.clf.fit(feats, self.df_train[target])

    def predict(self):
        """Predict the whole eval feature set"""
        predictions =  self.clf.predict(self.feats_test.df.to_numpy())
        report = Reporter(self.df_test[glob_conf.config['DATA']['target']].to_numpy().astype(float), predictions)
        return report
