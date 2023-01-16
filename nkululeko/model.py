# model.py
from nkululeko.util import Util 
import pandas as pd
import numpy as np
import nkululeko.glob_conf as glob_conf
import sklearn.utils
from nkululeko.reporter import Reporter
import ast
from sklearn.model_selection import GridSearchCV
import pickle
import random
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import StratifiedKFold


class Model:
    """Generic model class for linear (non-neural) algorithms"""

    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor taking the configuration and all dataframes"""
        self.df_train, self.df_test, self.feats_train, self.feats_test = df_train, df_test, feats_train, feats_test
        self.util = Util()
        self.target = self.util.config_val('DATA', 'target', 'emotion')
        self.run = 0
        self.epoch = 0
        self.loso = self.util.config_val('MODEL', 'loso', False)
        self.logo = self.util.config_val('MODEL', 'logo', False)
        self.xfoldx = self.util.config_val('MODEL', 'k_fold_cross', False)
        
    def set_testdata(self, data_df, feats_df):
        self.df_test, self.feats_test = data_df, feats_df

    def reset_test(self,  df_test, feats_test):
        self.df_test, self.feats_test = df_test, feats_test


    def set_id(self, run, epoch):
        self.run = run
        self.epoch = epoch
        dir = self.util.get_path('model_dir')
        name = f'{self.util.get_exp_name(only_train=True)}_{self.run}_{self.epoch:03d}.model'
        self.store_path = dir+name


    def _x_fold_cross(self):
        # ignore train and test sets and do a "leave one speaker out"  evaluation
        self.util.debug(f'ignoring splits and doing {self.xfoldx} fold cross validation')
        feats = self.feats_train.append(self.feats_test)
        annos = self.df_train.append(self.df_test)
        targets = annos[self.target]
        _skf = StratifiedKFold(n_splits=int(self.xfoldx))
        truths, preds, results = [], [], []
        g_index = 0
        # leave-one-speaker loop    
        for train_index, test_index in _skf.split(
            feats, 
            targets,
        ):
            train_x = feats.iloc[train_index].to_numpy()
            train_y = targets[train_index]
            self.clf.fit(train_x, train_y)
            truth_x = feats.iloc[test_index].to_numpy()
            truth_y = targets[test_index]
            predict_y = self.clf.predict(truth_x)
            report = Reporter(truth_y.astype(float), predict_y, self.run, self.epoch)
            self.util.debug(f'result for fold {g_index}: {report.get_result().get_test_result()} ')
            results.append(float(report.get_result().test))            
            truths.append(truth_y)
            preds.append(predict_y)
            g_index += 1
            
        # combine speaker folds
        truth = pd.concat(truths)
        truth.name = 'truth'
        pred = pd.Series(
            np.concatenate(preds),
            index=truth.index,
            name='prediction',
        )
        self.truths = truth
        self.preds = pred        
        results = np.asarray(results)
        self.util.debug(f'KFOLD: {self.xfoldx} folds: mean {results.mean():.3f}, std: {results.std():.3f}')

    def _loso(self):
        # ignore train and test sets and do a "leave one speaker out"  evaluation
        self.util.debug('ignoring splits and doing LOSO')
        feats = self.feats_train.append(self.feats_test)
        annos = self.df_train.append(self.df_test)
        targets = annos[self.target]
        _logo = LeaveOneGroupOut()
        truths, preds, results = [], [], []
        speakers = annos['speaker']
        g_index = 0
        # leave-one-speaker loop    
        for train_index, test_index in _logo.split(
            feats, 
            targets, 
            groups=speakers,
        ):
            train_x = feats.iloc[train_index].to_numpy()
            train_y = targets[train_index]
            self.clf.fit(train_x, train_y)
            
            truth_x = feats.iloc[test_index].to_numpy()
            truth_y = targets[test_index]
            predict_y = self.clf.predict(truth_x)
            report = Reporter(truth_y.astype(float), predict_y, self.run, self.epoch)
            self.util.debug(f'result for speaker {g_index}: {report.get_result().get_test_result()} ')
            truths.append(truth_y)
            preds.append(predict_y)
            g_index += 1
            results.append(float(report.get_result().test))            
        # combine speaker folds
        truth = pd.concat(truths)
        truth.name = 'truth'
        pred = pd.Series(
            np.concatenate(preds),
            index=truth.index,
            name='prediction',
        )
        self.truths = truth
        self.preds = pred        
        results = np.asarray(results)
        self.util.debug(f'LOSO: {self.loso} folds: mean {results.mean():.3f}, std: {results.std():.3f}')

    def _do_logo(self):
        # ignore train and test sets and do a "leave one speaker group out"  evaluation
        self.util.debug('ignoring splits and doing LOSGO')
        feats = self.feats_train.append(self.feats_test)
        annos = self.df_train.append(self.df_test)
        targets = annos[self.target]
        _logo = LeaveOneGroupOut()
        truths, preds, results = [], [], []
        # get unique list of speakers
        speakers = annos['speaker'].unique()
        # create a random dictionary of groups
        sdict = {}
        for i, s in enumerate(speakers):
            sdict[s] = random.sample(range(int(self.logo)), 1)[0]    
        # add this to the annotations  
        annos['speaker_groups'] = annos['speaker'].apply(lambda x: str(sdict[x]))
        speaker_groups = annos['speaker_groups']
        g_index = 0
        # leave-one-speaker loop    
        for train_index, test_index in _logo.split(
            feats, 
            targets, 
            groups=speaker_groups,
        ):
            train_x = feats.iloc[train_index].to_numpy()
            train_y = targets[train_index]
            self.clf.fit(train_x, train_y)
            
            truth_x = feats.iloc[test_index].to_numpy()
            truth_y = targets[test_index]
            predict_y = self.clf.predict(truth_x)
            report = Reporter(truth_y.astype(float), predict_y, self.run, self.epoch)
            result = report.get_result().get_test_result()
            self.util.debug(f'result for speaker group {g_index}: {result} ')
            results.append(float(report.get_result().test))
            truths.append(truth_y)
            preds.append(predict_y)
            g_index += 1
            
        # combine speaker folds
        truth = pd.concat(truths)
        truth.name = 'truth'
        pred = pd.Series(
            np.concatenate(preds),
            index=truth.index,
            name='prediction',
        )
        self.truths = truth
        self.preds = pred
        results = np.asarray(results)
        self.util.debug(f'LOGO: {self.logo} folds: mean {results.mean():.3f}, std: {results.std():.3f}')

    def train(self):
        """Train the model"""
        # # first check if the model already has been trained
        # if os.path.isfile(self.store_path):
        #     self.load(self.run, self.epoch)
        #     self.util.debug(f'reusing model: {self.store_path}')
        #     return

        # first check if leave on  speaker out is wanted
        if self.loso:
            self._loso()
            return
        # then if leave one speaker group out validation is wanted
        if self.logo:
            self._do_logo()
            return
        # then if x fold cross validation is wanted
        if self.xfoldx:
            self._x_fold_cross()
            return

        # check for NANs in the features
        # set up the data_loaders
        if self.feats_train.isna().to_numpy().any():
            self.util.debug(f'Model, train: replacing {self.feats_train.isna().sum().sum()} NANs with 0')
            self.feats_train = self.feats_train.fillna(0)
        # remove labels from features
        feats = self.feats_train.to_numpy()
        # compute class weights
        if self.util.config_val('MODEL', 'class_weight', False):      
            self.classes_weights = sklearn.utils.class_weight.compute_sample_weight(
                class_weight='balanced',
                y=self.df_train[self.target]
            )

        tuning_params = self.util.config_val('MODEL', 'tuning_params', False)
        if tuning_params:
            # tune the model meta parameters
            tuning_params = ast.literal_eval(tuning_params)
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
                class_weight = self.util.config_val('MODEL', 'class_weight', False)
                if class_weight:
                    self.util.debug('using class weight')
                    self.clf.fit(feats, self.df_train[self.target], sample_weight=self.classes_weights)
                else:
                    self.clf.fit(feats, self.df_train[self.target])    
            except KeyError:
                self.clf.fit(feats, self.df_train[self.target])
            self.util.debug(f'winner parameters: {self.clf.best_params_}')
        else:
            class_weight = self.util.config_val('MODEL', 'class_weight', False)
            if class_weight:
                self.util.debug('using class weight')
                self.clf.fit(feats, self.df_train[self.target], sample_weight=self.classes_weights)
            else:
                labels = self.df_train[self.target]
                self.clf.fit(feats, labels)

    def get_predictions(self):
        predictions =  self.clf.predict(self.feats_test.to_numpy())
        return predictions

    def predict(self):
        if self.feats_test.isna().to_numpy().any():
            self.util.debug(f'Model, test: replacing {self.feats_test.isna().sum().sum()} NANs with 0')
            self.feats_test = self.feats_test.fillna(0)
        if self.loso or self.logo or self.xfoldx:
            report = Reporter(self.truths.astype(float), self.preds, self.run, self.epoch)
            return report
        """Predict the whole eval feature set"""
        predictions = self.get_predictions()
        report = Reporter(self.df_test[self.target]\
            .to_numpy().astype(float), predictions, self.run, self.epoch)
        return report

    def predict_sample(self, features):
        """Predict one sample"""
        prediction = {}
        # get the class probabilities
        predictions = self.clf.predict_proba(features)
        # pred = self.clf.predict(features)
        for i in range(len(self.clf.classes_)):
            cat = self.clf.classes_[i]
            prediction[cat] = predictions[0][i]
        return prediction


    def store(self):
        with open(self.store_path, 'wb') as handle:
            pickle.dump(self.clf, handle)

        
    def load(self, run, epoch):
        self.set_id(run, epoch)
        dir = self.util.get_path('model_dir')
        name = f'{self.util.get_exp_name(only_train=True)}_{self.run}_{self.epoch:03d}.model'
        with open(dir+name, 'rb') as handle:
            self.clf = pickle.load(handle)

    def load_path(self, path, run, epoch):
        self.set_id(run, epoch)
        with open(path, 'rb') as handle:
            self.clf = pickle.load(handle)