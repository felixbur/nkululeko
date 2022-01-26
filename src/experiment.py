from cgi import test
import numpy
from dataset import Dataset
from dataset_csv import Dataset_CSV
from dataset_ravdess import Ravdess
from feats_opensmile import Opensmileset
from runmanager import Runmanager
from test_predictor import Test_predictor
from util import Util
import glob_conf
from plots import Plots
from demo_predictor import Demo_predictor
import ast # To convert strings to objects
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scaler import Scaler
import pickle


class Experiment:
    """Main class specifying an experiment"""
    

    def __init__(self, config_obj):
        """
        Parameters
        ----------
        config_obj : a config parser object that sets the experiment parameters and being set as a global object.
        """

        self.set_globals(config_obj)
        self.name = glob_conf.config['EXP']['name']
        self.util = Util()

    def set_globals(self, config_obj):
        """install a config object in the global space"""
        glob_conf.init_config(config_obj)

    def load_datasets(self):
        """Load all databases specified in the configuration and map the labels"""
        ds = ast.literal_eval(glob_conf.config['DATA']['databases'])
        self.datasets = {}
        for d in ds:
            if d == 'ravdess':
                data = Ravdess()
            else:
                ds_type = self.util.config_val('DATA', d+'.type', 'audformat')
                if ds_type == 'audformat':
                    data = Dataset(d)
                elif ds_type == 'csv':
                    data = Dataset_CSV(d)
                else:
                    self.util.error(f'unknown data type: {ds_type}')
            data.load()
            self.datasets.update({d: data})
        self.target = self.util.config_val('DATA', 'target', 'emotion')

    def fill_train_and_tests(self):
        """Set up train and development sets. The method should be specified in the config."""
        self.df_train, self.df_test = pd.DataFrame(), pd.DataFrame()
        strategy = self.util.config_val('DATA', 'strategy', 'train_test')
        # some datasets against others in their entirety
        if strategy == 'cross_data':
            train_dbs = ast.literal_eval(glob_conf.config['DATA']['trains'])
            test_dbs = ast.literal_eval(glob_conf.config['DATA']['tests'])
            for dn in train_dbs:
                d = self.datasets[dn]
                d.prepare_labels()
                self.df_train = self.df_train.append(d.df)
            for dn in test_dbs:
                d = self.datasets[dn]
                d.prepare_labels()
                self.df_test = self.df_test.append(d.df)
        elif strategy == 'train_test':
            # default: train vs. test combined from all datasets
            for d in self.datasets.values():
                d.split()
                d.prepare_labels()
                self.df_train = self.df_train.append(d.df_train)
                self.df_test = self.df_test.append(d.df_test)
        else:
            self.util.error(f'unknown strategy: {strategy}')

        # encode the labels
        if self.util.exp_is_classification():
            datatype = self.util.config_val('DATA', 'type', 'dummy')
            if datatype == 'continuous':
                test_cats = self.df_test['class_label'].unique()
                train_cats = self.df_train['class_label'].unique()
            else:
                test_cats = self.df_test[self.target].unique()
                train_cats = self.df_train[self.target].unique()
            if type(test_cats) == numpy.ndarray:
                self.util.debug(f'Categories test: {test_cats}')
            else:
                self.util.debug(f'Categories test: {test_cats.to_list()}')
            if type(train_cats) == numpy.ndarray:
                self.util.debug(f'Categories train: {train_cats}')
            else:
                self.util.debug(f'Categories train: {train_cats.to_list()}')

            # encode the labels as numbers
            self.label_encoder = LabelEncoder()
            self.df_train[self.target] = self.label_encoder.fit_transform(self.df_train[self.target])
            self.df_test[self.target] = self.label_encoder.transform(self.df_test[self.target])
            glob_conf.set_label_encoder(self.label_encoder)
        else:
            pass
        self.util.debug(f'{self.df_test.speaker.nunique()} speakers in test and {self.df_train.speaker.nunique()} speakers in train')
        augment = self.util.config_val('DATA', 'augment', 0)
        if augment:
            self.augment_train()
        if self.util.config_val('PLOT', 'value_counts', False):
            self.plot_distribution()


    def plot_distribution(self):
        """Plot the distribution of samples and speaker per target class and biological sex"""
        from plots import Plots
        plot = Plots()
        if self.util.exp_is_classification():
            self.df_train['labels'] = self.label_encoder.inverse_transform(self.df_train[self.target])
            self.df_test['labels'] = self.label_encoder.inverse_transform(self.df_test[self.target])
            plot.describe_df('dev_set', self.df_test, 'labels', f'test_distplot.png')
            plot.describe_df('train_set', self.df_train, 'labels', f'train_distplot.png')
        else:
            plot.describe_df('dev_set', self.df_test, self.target, f'test_distplot.png')
            plot.describe_df('train_set', self.df_train, self.target, f'train_distplot.png')


    def augment_train(self):
        """Augment the train dataframe"""
        from augmenter import Augmenter
        augment_train = Augmenter(self.df_train)
        df_train_aug = augment_train.augment()
        self.df_train = self.df_train.append(df_train_aug)


    def extract_feats(self):
        """Extract the features for train and dev sets. 
        
        They will be stored on disk and need to be removed manually.
        
        The string FEATS.feats_type is read from the config, defaults to os. 
        
        """
        df_train, df_test = self.df_train, self.df_test
        strategy = self.util.config_val('DATA', 'strategy', 'train_test')
        feats_type = self.util.config_val('FEATS', 'type', 'os')
        feats_name = "_".join(ast.literal_eval(glob_conf.config['DATA']['databases']))
        feats_name = f'{feats_name}_{strategy}_{feats_type}'   
        scale = 1
        if feats_type=='os':
            self.feats_train = Opensmileset(f'{feats_name}_train', df_train)
            self.feats_train.extract()
            self.feats_train.filter()
            self.feats_test = Opensmileset(f'{feats_name}_test', df_test)
            self.feats_test.extract()
            self.feats_test.filter()
            self.util.debug(f'train shape : {self.feats_train.df.shape}, test shape:{self.feats_test.df.shape}')
        elif feats_type=='audid':
            from feats_audid import AudIDset
            self.feats_train = AudIDset(f'{feats_name}_train', df_train)
            self.feats_train.extract()
            self.feats_train.filter()
            self.feats_test = AudIDset(f'{feats_name}_test', df_test)
            self.feats_test.extract()
            self.feats_test.filter()
        elif feats_type=='trill':
            from feats_trill import TRILLset
            self.feats_train = TRILLset(f'{feats_name}_train', df_train)
            self.feats_train.extract()
            self.feats_train.filter()
            self.feats_test = TRILLset(f'{feats_name}_test', df_test)
            self.feats_test.extract()
            self.feats_test.filter()
        elif feats_type=='wav2vec':
            from feats_wav2vec2 import Wav2vec2
            self.feats_train = Wav2vec2(f'{feats_name}_train', df_train)
            self.feats_train.extract()
            self.feats_train.filter()
            self.feats_test = Wav2vec2(f'{feats_name}_test', df_test)
            self.feats_test.extract()
            self.feats_test.filter()
        elif feats_type=='mld':
            from feats_mld import MLD_set
            self.feats_train = MLD_set(f'{feats_name}_train', df_train)
            self.feats_train.extract()
            self.feats_train.filter()
            self.feats_test = MLD_set(f'{feats_name}_test', df_test)
            self.feats_test.extract()
            self.feats_test.filter()
            # remove samples that were not extracted by MLD
            self.df_test = self.df_test.loc[self.df_test.index.intersection(self.feats_test.df.index)]
            self.df_train = self.df_train.loc[self.df_train.index.intersection(self.feats_train.df.index)]
            if self.feats_train.df.isna().to_numpy().any():
                self.util.error('exp 2: NANs exist')
        elif feats_type=='xbow':
            from feats_oxbow import Openxbow
            self.feats_train = Openxbow(f'{feats_name}_train', df_train, is_train=True)
            self.feats_train.extract()
            self.feats_train.filter()
            self.feats_test = Openxbow(f'{feats_name}_test', df_test)
            self.feats_test.extract()
            self.feats_test.filter()
            self.util.debug(f'train shape : {self.feats_train.df.shape}, test shape:{self.feats_test.df.shape}')

        elif feats_type=='spectra':
            # compute the spectrograms
            from feats_spectra import Spectraloader # not yet open source
            test_specs = Spectraloader(f'{feats_name}_test', df_test)
            test_specs.make_feats()
            self.feats_test = test_specs.get_loader()
            self.feats_train = None
            if df_train.shape[0]>0:
                train_specs = Spectraloader(f'{feats_name}_train', df_train)
                train_specs.make_feats()
                self.feats_train = train_specs.get_loader()
            scale = 0
        else:
            self.util.error(f'unknown feats_type: {feats_type}')

        if scale:
            self.scale()

        # check if a tsne should be plotted
        tsne = self.util.config_val('PLOT', 'tsne', False)
        if tsne and self.util.exp_is_classification():
            from plots import Plots
            plots = Plots()
            plots.plotTsne(self.feats_train.df, self.df_train['class_label'], self.util.get_exp_name()+'_tsne')


    def scale(self):
        try:
            dummy = glob_conf.config['FEATS']['scale'] 
            self.scaler = Scaler(self.df_train, self.df_test, self.feats_train, self.feats_test)
            self.feats_train.df, self.feats_test.df = self.scaler.scale()
        except KeyError:
            pass

    def init_runmanager(self):
        """Initialize the manager object for the runs."""
        self.runmgr = Runmanager(self.df_train, self.df_test, self.feats_train, self.feats_test)

    def run(self):
        """Do the runs."""
        self.runmgr.do_runs()

        # access the results of the last run
        self.reports = self.runmgr.best_results
        # try to save yourself
        save = self.util.config_val('EXP', 'save', False)
        if save: 
            # save the experiment for future use
            self.save(self.util.get_save_name())

        # self.__collect_reports()
        self.util.print_best_results(self.reports)

        # check if the test predictions should be saved to disk
        test_pred_file = self.util.config_val('EXP', 'save_test', 'False')
        if test_pred_file:
            self.predict_test_and_save(test_pred_file)

        return self.reports    


    def print_best_model(self):
        self.runmgr.print_best_result_runs()

    def __collect_reports(self):
        self.results, self.losses, self.train_results = [], [], []
        for r in self.reports:
            self.results.append(r.get_result().test)
            self.losses.append(r.get_result().loss)
            self.train_results.append(r.get_result().train)

    def demo(self):
        model = self.runmgr.get_best_model()
        feature_extractor = self.feats_train
        demo = Demo_predictor(model, feature_extractor, self.label_encoder)
        demo.run_demo()

    def  predict_test_and_save(self, name):
        model = self.runmgr.get_best_model()
        test_predictor = Test_predictor(model, self.df_test, self.label_encoder, name)        
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
        except AttributeError: 
            self.util.error('Save experiment: Can\'t pickle local object')