import random
import os
import time
import numpy as np
from nkululeko.dataset import Dataset
from nkululeko.dataset_csv import Dataset_CSV
from nkululeko.dataset_ravdess import Ravdess
from nkululeko.filter_data import filter_min_dur
from nkululeko.runmanager import Runmanager
from nkululeko.test_predictor import Test_predictor
from nkululeko.feats_analyser import FeatureAnalyser
from nkululeko.util import Util
from nkululeko.feature_extractor import FeatureExtractor
from nkululeko.plots import Plots
import nkululeko.glob_conf as glob_conf
from nkululeko.demo_predictor import Demo_predictor
import ast # To convert strings to objects
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nkululeko.scaler import Scaler
import pickle
import audformat

class Experiment:
    """Main class specifying an experiment
    
    """
    

    def __init__(self, config_obj):
        """
        Parameters
        ----------
        config_obj : a config parser object that sets the experiment parameters and being set as a global object.
        """

        self.set_globals(config_obj)
        self.name = glob_conf.config['EXP']['name']
        self.util = Util()
        glob_conf.set_util(self.util)
        self.loso = self.util.config_val('MODEL', 'loso', False)
        self.logo = self.util.config_val('MODEL', 'logo', False)
        self.xfoldx = self.util.config_val('MODEL', 'k_fold_cross', False)
        self.start = time.process_time()

    def get_name(self):
        return self.util.get_exp_name()

    def set_globals(self, config_obj):
        """install a config object in the global space"""
        glob_conf.init_config(config_obj)

    def load_datasets(self):
        """Load all databases specified in the configuration and map the labels"""
        ds = ast.literal_eval(glob_conf.config['DATA']['databases'])
        self.datasets = {}
        self.got_speaker, self.got_gender, self.got_age = False, False, False
        for d in ds:
            # if d == 'ravdess':
            #     data = Ravdess()
            # else:
            ds_type = self.util.config_val_data(d, 'type', 'audformat')
            if ds_type == 'audformat':
                data = Dataset(d)
            elif ds_type == 'csv':
                data = Dataset_CSV(d)
            else:
                self.util.error(f'unknown data type: {ds_type}')
            data.load()
            data.prepare()
            if data.got_gender:
                self.got_gender = True
            if data.got_age:
                self.got_age = True
            if data.got_speaker:
                self.got_speaker = True
            self.datasets.update({d: data})
        self.target = self.util.config_val('DATA', 'target', 'emotion')

    def _import_csv(self, storage):
        # df = pd.read_csv(storage, header=0, index_col=[0,1,2])
        # df.index.set_levels(pd.to_timedelta(df.index.levels[1]), level=1)
        # df.index.set_levels(pd.to_timedelta(df.index.levels[2]), level=2)
        df = audformat.utils.read_csv(storage)
        df.is_labeled = True if self.target in df else False
        return df

    def fill_tests(self):
        """Only fill a new test set"""

        test_dbs = ast.literal_eval(glob_conf.config['DATA']['tests'])
        self.df_test = pd.DataFrame()
        start_fresh = eval(self.util.config_val('DATA', 'no_reuse', 'False'))
        store = self.util.get_path('store')
        storage_test = f'{store}extra_testdf.csv'
        if os.path.isfile(storage_test) \
            and not start_fresh:
            self.util.debug(f'reusing previously stored {storage_test}')
            self.df_test = self._import_csv(storage_test)
        else:
            for d in test_dbs:
                if d == 'ravdess':
                    data = Ravdess()
                else:
                    ds_type = self.util.config_val_data(d, 'type', 'audformat')
                    if ds_type == 'audformat':
                        data = Dataset(d)
                    elif ds_type == 'csv':
                        data = Dataset_CSV(d)
                    else:
                        self.util.error(f'unknown data type: {ds_type}')
                data.load()
                if data.got_gender:
                    self.got_gender = True
                if data.got_age:
                    self.got_age = True
                if data.got_speaker:
                    self.got_speaker = True
                data.prepare_labels()
                self.df_test = pd.concat([self.df_test, self.util.make_segmented_index(data.df)])
                self.df_test.is_labeled = data.is_labeled
            self.df_test.got_gender = self.got_gender
            self.df_test.got_speaker = self.got_speaker
            # self.util.set_config_val('FEATS', 'needs_features_extraction', 'True')
            # self.util.set_config_val('FEATS', 'no_reuse', 'True')
            self.df_test['class_labels'] = self.df_test[self.target]
            self.df_test[self.target] = self.label_encoder.transform(self.df_test[self.target])
            self.df_test.to_csv(storage_test)

    def fill_train_and_tests(self):
        """Set up train and development sets. The method should be specified in the config."""
        store = self.util.get_path('store')
        storage_test = f'{store}testdf.csv'
        storage_train = f'{store}traindf.csv'
        start_fresh = eval(self.util.config_val('DATA', 'no_reuse', 'False'))
        if os.path.isfile(storage_train) and os.path.isfile(storage_test) \
            and not start_fresh:
            self.util.debug(f'reusing previously stored {storage_test} and {storage_train}')
            self.df_test = self._import_csv(storage_test)
            self.df_train = self._import_csv(storage_train)
        else:
            self.df_train, self.df_test = pd.DataFrame(), pd.DataFrame()
            strategy = self.util.config_val('DATA', 'strategy', 'traintest')
            # some datasets against others in their entierty
            if strategy == 'cross_data':
                train_dbs = ast.literal_eval(glob_conf.config['DATA']['trains'])
                test_dbs = ast.literal_eval(glob_conf.config['DATA']['tests'])
                for dn in train_dbs:
                    d = self.datasets[dn]
                    d.prepare_labels()
                    self.df_train = self.df_train.append(self.util.make_segmented_index(d.df))
                    self.df_train.is_labeled = d.is_labeled
                for dn in test_dbs:
                    d = self.datasets[dn]
                    d.prepare_labels()
                    self.df_test = self.df_test.append(self.util.make_segmented_index(d.df))
                    self.df_test.is_labeled = d.is_labeled
            elif strategy == 'traintest':
                # default: train vs. test combined from all datasets
                for d in self.datasets.values():
                    d.split()
                    d.prepare_labels()
                    self.df_train = pd.concat([self.df_train, self.util.make_segmented_index(d.df_train)])
                    self.df_train.is_labeled = d.is_labeled
                    self.df_test = pd.concat([self.df_test, self.util.make_segmented_index(d.df_test)])
                    self.df_test.is_labeled = d.is_labeled
            else:
                self.util.error(f'unknown strategy: {strategy}')
            # save the file lists to disk for later reuse
            store = self.util.get_path('store')
            storage_test = f'{store}testdf.csv'
            storage_train = f'{store}traindf.csv'
            self.df_test.to_csv(storage_test)
            self.df_train.to_csv(storage_train)

        self.df_train.got_gender = self.got_gender
        self.df_train.got_age = self.got_age
        self.df_train.got_speaker = self.got_speaker
        self.df_test.got_gender = self.got_gender
        self.df_test.got_speaker = self.got_speaker
        self.df_test.got_age = self.got_age

        # Check for filters
        min_dur_test = self.util.config_val('DATA', 'min_dur_test', False)
        if min_dur_test:
            pre = self.df_test.shape[0]
            self.df_test = filter_min_dur(self.df_test, min_dur_test)
            post = self.df_test.shape[0]
            self.util.debug(f'Dropped {pre-post} test samples shorter than {min_dur_test} seconds (from {pre} to {post} samples)')

        # encode the labels
        if self.util.exp_is_classification():
            datatype = self.util.config_val('DATA', 'type', 'dummy')
            if datatype == 'continuous':
                if self.df_test.is_labeled:
                    # remember the target in case they get labelencoded later
                    self.df_test['class_label'] = self.df_test[self.target]
                    test_cats = self.df_test['class_label'].unique()
                else:
                    # if there is no target, copy a dummy label
                    self.df_test = self._add_random_target(self.df_test)
                if self.df_train.is_labeled:
                    # remember the target in case they get labelencoded later
                    self.df_train['class_label'] = self.df_train[self.target]
                    train_cats = self.df_train['class_label'].unique()
            else:
                if self.df_test.is_labeled:
                    test_cats = self.df_test[self.target].unique()
                else:   
                    # if there is no target, copy a dummy label
                    self.df_test = self._add_random_target(self.df_test)
                train_cats = self.df_train[self.target].unique()
            if self.df_test.is_labeled:
                if type(test_cats) == np.ndarray:
                    self.util.debug(f'Categories test: {test_cats}')
                else:
                    self.util.debug(f'Categories test: {test_cats.to_list()}')
            if type(train_cats) == np.ndarray:
                self.util.debug(f'Categories train: {train_cats}')
            else:
                self.util.debug(f'Categories train: {train_cats.to_list()}')

            # encode the labels as numbers
            self.label_encoder = LabelEncoder()
            self.df_train[self.target] = self.label_encoder.fit_transform(self.df_train[self.target])
            self.df_test[self.target] = self.label_encoder.transform(self.df_test[self.target])
            glob_conf.set_label_encoder(self.label_encoder)
        if self.got_speaker:
            self.util.debug(f'{self.df_test.speaker.nunique()} speakers in test and {self.df_train.speaker.nunique()} speakers in train')

        target_factor = self.util.config_val('DATA', 'target_divide_by', False)
        if target_factor:
            self.df_test[self.target] = self.df_test[self.target] / float(target_factor)
            self.df_train[self.target] = self.df_train[self.target] / float(target_factor)
            if not self.util.exp_is_classification():
                self.df_test['class_label'] = self.df_test['class_label'] / float(target_factor)
                self.df_train['class_label'] = self.df_train['class_label'] / float(target_factor)

    def _add_random_target(self, df):
        labels = self.util.get_labels()
        a = [None]*len(df)
        for i in range(0, len(df)):
            a[i] = random.choice(labels)
        df[self.target] = a
        return df

    def plot_distribution(self):
        """Plot the distribution of samples and speaker per target class and biological sex"""
        plot = Plots()
        if self.util.exp_is_classification():
            # self.df_train['labels'] = self.label_encoder.inverse_transform(self.df_train[self.target])
            # if self.df_test.is_labeled:
            #     self.df_test['labels'] = self.label_encoder.inverse_transform(self.df_test[self.target])
            if self.df_test.shape[0] > 0:
                plot.describe_df('dev_set', self.df_test, self.target, f'test_distplot')
            plot.describe_df('train_set', self.df_train, self.target, f'train_distplot')
        else:
            if self.df_test.shape[0] > 0:
                plot.describe_df('dev_set', self.df_test, self.target, f'test_distplot')
            plot.describe_df('train_set', self.df_train, self.target, f'train_distplot')

    def extract_test_feats(self):
        self.feats_test = pd.DataFrame()
        feats_name = "_".join(ast.literal_eval(glob_conf.config['DATA']['tests']))
        self.feature_extractor = FeatureExtractor() 
        self.feature_extractor.set_data(self.df_test, feats_name, 'test')
        self.feats_test = self.feature_extractor.extract()
        self.util.debug(f'Test features shape:{self.feats_test.shape}')

    def extract_feats(self):
        """Extract the features for train and dev sets. 
        
        They will be stored on disk and need to be removed manually.
        
        The string FEATS.feats_type is read from the config, defaults to os. 
        
        """
        df_train, df_test = self.df_train, self.df_test
        feats_name = "_".join(ast.literal_eval(glob_conf.config['DATA']['databases']))
        self.feats_test, self.feats_train = pd.DataFrame(), pd.DataFrame()
        self.feature_extractor = FeatureExtractor() 
        self.feature_extractor.set_data(df_train, feats_name, 'train')
        self.feats_train =self.feature_extractor.extract()
        self.feature_extractor.set_data(df_test, feats_name, 'test')
        self.feats_test = self.feature_extractor.extract()
        self.util.debug(f'All features: train shape : {self.feats_train.shape}, test shape:{self.feats_test.shape}')

        self._check_scale()

    def augment(self):
        """
        Augment the training set
        """
        from nkululeko.augmenter import Augmenter
        augmenter = Augmenter(self.df_train)
        augmenter.augment()

    # def augment_train(self):
    #     """Augment the train dataframe"""
    #     from nkululeko.augmenter import Augmenter
    #     augment_train = Augmenter(self.df_train)
    #     df_train_aug = augment_train.augment()
    #     self.df_train = self.df_train.append(df_train_aug)


    def analyse_features(self):
        """
        Do a feature exploration
        
        """

        if self.util.config_val('EXPL', 'value_counts', False):
            self.plot_distribution()

        sample_selection = self.util.config_val('EXPL', 'sample_selection', 'False')
        if sample_selection=='all':
            df_feats = pd.concat([self.feats_train, self.feats_test])
            df_labels = pd.concat([self.df_train, self.df_test])
        elif sample_selection=='train':
            df_feats = self.feats_train
            df_labels = self.df_train
        elif sample_selection=='test':
            df_feats = self.feats_test
            df_labels = self.df_test
        elif sample_selection=='False':
            pass
        else:
            self.util.error(f'unkown feature_distribution specifier {sample_selection}, should be [all | train | test]')
        if sample_selection in ('all', 'train', 'test'):
            feat_analyser = FeatureAnalyser(sample_selection, df_labels, df_feats)        
            feat_analyser.analyse()

        # check if a scatterplot should be done
        scatter_var = eval(self.util.config_val('EXPL', 'scatter', 'False'))
        if scatter_var:
            scatters = ast.literal_eval(glob_conf.config['EXPL']['scatter'])
            if self.util.exp_is_classification():
                plots = Plots()
                all_feats =self.feats_train.append(self.feats_test)
                all_labels = self.df_train['class_label'].append(self.df_test['class_label'])
                for scatter in scatters:
                    plots.scatter_plot(all_feats, all_labels, scatter)
            else:
                 self.util.debug('can\'t do scatterplot if not classification')


        # check if a tsne should be plotted
        tsne = eval(self.util.config_val('EXPL', 'tsne', 'False'))
        if tsne: 
            if self.util.exp_is_classification():
                plots = Plots()
                all_feats =self.feats_train.append(self.feats_test)
                all_labels = self.df_train['class_label'].append(self.df_test['class_label'])
                plots.plotTsne(all_feats, all_labels, self.util.get_exp_name()+'_tsne')
            else:
                 self.util.debug('can\'t plot tsne if not classification')


    def _check_scale(self):
        scale = self.util.config_val('FEATS', 'scale', False)
        if scale: 
            self.scaler = Scaler(self.df_train, self.df_test, self.feats_train, self.feats_test, scale)
            self.feats_train, self.feats_test = self.scaler.scale()

    def init_runmanager(self):
        """Initialize the manager object for the runs."""
        self.runmgr = Runmanager(self.df_train, self.df_test, self.feats_train, self.feats_test)

    def run(self):
        """Do the runs."""
        self.runmgr.do_runs()

        # access the best results all runs
        self.reports = self.runmgr.best_results
        # try to save yourself
        save = self.util.config_val('EXP', 'save', False)
        if save: 
            # save the experiment for future use
            self.save(self.util.get_save_name())

        # self.__collect_reports()
        self.util.print_best_results(self.reports)

        # check if the test predictions should be saved to disk
        test_pred_file = self.util.config_val('EXP', 'save_test', False)
        if test_pred_file:
            self.predict_test_and_save(test_pred_file)

        # check if the majority voting for all speakers should be plotted
        conf_mat_per_speaker_function = self.util.config_val('PLOT', 'combine_per_speaker', False)
        if (conf_mat_per_speaker_function):
            self.plot_confmat_per_speaker(conf_mat_per_speaker_function)
        used_time = time.process_time() - self.start
        self.util.debug(f'Done, used {used_time:.3f} seconds')

        # check if a test set should be labeled by the model:
        label_data = self.util.config_val('DATA', 'label_data', False)
        label_result = self.util.config_val('DATA', 'label_result', False)
        if label_data and label_result:
            self.predict_test_and_save(label_result)

        return self.reports    

    def plot_confmat_per_speaker(self, function):
        if self.loso or self.logo or self.xfoldx:
            self.util.debug('plot combined speaker predictions not possible for cross validation')
            return
        best = self._get_best_report(self.reports)
        # if not best.is_classification:
        #     best.continuous_to_categorical()
        truths = best.truths
        preds = best.preds
        speakers = self.df_test.speaker.values
        print(f'{len(truths)} {len(preds)} {len(speakers) }')
        df = pd.DataFrame(data={'truth':truths, 'pred':preds, 'speaker':speakers})
        plot_name = 'result_combined_per_speaker'
        self.util.debug(f'plotting speaker combination ({function}) confusion matrix to {plot_name}')
        best.plot_per_speaker(df, plot_name, function)

    def _get_best_report(self, best_reports):
        return self.runmgr.get_best_result(best_reports)

    def print_best_model(self):
        self.runmgr.print_best_result_runs()

    def demo(self, file, is_list):
        model = self.runmgr.get_best_model()
        labelEncoder = None
        try: 
            labelEncoder = self.label_encoder
        except AttributeError:
            pass
        demo = Demo_predictor(model, file, is_list, self.feature_extractor, labelEncoder)
        demo.run_demo()

    def predict_test_and_save(self, result_name):
        model = self.runmgr.get_best_model()
        model.set_testdata(self.df_test, self.feats_test)
        test_predictor = Test_predictor(model, self.df_test, self.label_encoder, result_name)        
        test_predictor.predict_and_store()


    def load(self, filename):
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          
        self.__dict__.update(tmp_dict) 

    def save(self, filename):
        try:
            f = open(filename, 'wb')
            pickle.dump(self.__dict__, f)
            f.close()
        except (AttributeError, TypeError) as error: 
            self.util.debug(f'Save experiment: Can\'t pickle local object: {error}')