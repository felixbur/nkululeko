# model.py
from util import Util 
import glob_conf
import sklearn.utils
from reporter import Reporter
import ast
from sklearn.model_selection import GridSearchCV
import pickle

class Model:
    """Generic model class for linear (non-neural) algorithms"""

    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor taking the configuration and all dataframes"""
        self.df_train, self.df_test, self.feats_train, self.feats_test = df_train, df_test, feats_train, feats_test
        self.util = Util()
        target = glob_conf.config['DATA']['target']
        self.run = 0
        self.epoch = 0

    def set_id(self, run, epoch):
        self.run = run
        self.epoch = epoch
 

    def train(self):
        """Train the model"""
        target = glob_conf.config['DATA']['target']
        # check for NANs in the features
        if self.feats_train.df.isna().to_numpy().any():
            self.util.error('can\'t train: NANs exist')
        # remove labels from features
        feats = self.feats_train.df.to_numpy()
        # compute class weights
        if self.util.config_val('MODEL', 'class_weight', 0):      
            self.classes_weights = sklearn.utils.class_weight.compute_sample_weight(
                class_weight='balanced',
                y=self.df_train[target]
            )

        try:
            # tune the model meta parameters
            tuning_params = ast.literal_eval(glob_conf.config['MODEL']['tuning_params'])
            tuned_params={}
            try:
                scoring = glob_conf.config['MODEL']['scoring']
            except KeyError:
                self.util.error('got tuning params but no scoring')
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
                else:
                    self.clf.fit(feats, self.df_train[target])    
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

    def get_predictions(self):
        predictions =  self.clf.predict(self.feats_test.df.to_numpy())
        return predictions

    def predict(self):
        """Predict the whole eval feature set"""
        predictions = self.get_predictions()
        report = Reporter(self.df_test[glob_conf.config['DATA']['target']]\
            .to_numpy().astype(float), predictions, self.run, self.epoch)
        return report

    def predict_sample(self, features):
        """Predict one sample"""
        prediction = {}
        predictions = self.clf.predict_proba(features)
        pred = self.clf.predict(features)
        for i in range(len(self.clf.classes_)):
            cat = self.clf.classes_[i]
            prediction[cat] = predictions[0][i]
        return prediction


    def store(self):
        dir = self.util.get_path('model_dir')
        name = f'{self.util.get_exp_name()}_{self.run}_{self.epoch:03d}.model'
        with open(dir+name, 'wb') as handle:
            pickle.dump(self.clf, handle)

        
    def load(self, run, epoch):
        self.set_id(run, epoch)
        dir = self.util.get_path('model_dir')
        name = f'{self.util.get_exp_name()}_{self.run}_{self.epoch:03d}.model'
        with open(dir+name, 'rb') as handle:
            self.clf = pickle.load(handle)