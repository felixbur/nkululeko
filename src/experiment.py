from opensmile.core.define import FeatureSet
from dataset import Dataset
from emodb import Emodb
from feats_opensmile import Opensmileset
from runmanager import Runmanager
from util import Util
import glob_conf
import ast # To convert strings to objects
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from feats_spectra import Spectraloader
from scaler import Scaler


class Experiment:
    """Main class specifying an experiment"""
    
    name = '' # A name for the experiment. Will be used to label result and temporary files.
    datasets = {} # Dictionary to store the datasets that are loaded
    df_train = None # The training dataframe
    df_test = None # The evaluation dataframe
    feats_train = None # The training features
    feats_test = None # The test features
    runmgr = None  # The manager object for the runs
    labels = None # set of string values for the categories
    values = None # set of numerical values encoding the classes 

    def __init__(self, config_obj):
        """Constructor: takes a name and the config object"""
        glob_conf.init_config(config_obj)
        self.name = glob_conf.config['EXP']['name']
        self.util = Util()

    def load_datasets(self):
        """Load all databases specified in the configuration and map the labels"""
        ds = ast.literal_eval(glob_conf.config['DATA']['databases'])
        self.datasets = {}
        for d in ds:
            if d == 'emodb':
                data = Emodb()
            else:
                data = Dataset(d)
            data.load()
            data.prepare_labels()
            self.datasets.update({d: data})

    def fill_train_and_tests(self):
        """Set up train and development sets. The method should be specified in the config."""
        self.df_train, self.df_test = pd.DataFrame(), pd.DataFrame()
        strategy = self.util.config_val('DATA', 'strategy', 'train_test')
        # some datasets against others in their entirety
        if strategy == 'cross_data':
            train_dbs = ast.literal_eval(glob_conf.config['DATA']['trains'])
            test_dbs = ast.literal_eval(glob_conf.config['DATA']['tests'])
            for d in train_dbs:
                self.df_train = self.df_train.append(self.datasets[d].df)
            for d in test_dbs:
                self.df_test = self.df_test.append(self.datasets[d].df)
        elif strategy == 'train_test':
            # default: train vs. test combined from all datasets
            for d in self.datasets.values():
                d.split()
                self.df_train = self.df_train.append(d.df_train)
                self.df_test = self.df_test.append(d.df_test)
        else:
            print(f'unknown strategy: {strategy}')
            quit()

        # encode the labels
        if self.util.exp_is_classification():
            # encode the labels as numbers
            target = glob_conf.config['DATA']['target']
            self.label_encoder = LabelEncoder()
            self.df_train[target] = self.label_encoder.fit_transform(self.df_train[target])
            self.df_test[target] = self.label_encoder.transform(self.df_test[target])
            glob_conf.set_label_encoder(self.label_encoder)
        else:
            pass

    def extract_feats(self):
        """Extract the features for train and dev sets. They will be stored on disk and need to be removed manually."""
        df_train, df_test = self.df_train, self.df_test
        strategy = self.util.config_val('DATA', 'strategy', 'train_test')
        feats_type = self.util.config_val('FEATS', 'type', 'os')
        feats_name = "_".join(ast.literal_eval(glob_conf.config['DATA']['databases']))
        feats_name = f'{feats_name}_{strategy}_{feats_type}'   
        if feats_type=='os':
            self.feats_train = Opensmileset(f'{feats_name}_train', df_train)
            self.feats_train.extract()
            self.feats_train.filter()
            self.feats_test = Opensmileset(f'{feats_name}_test', df_test)
            self.feats_test.extract()
            self.feats_test.filter()
        elif feats_type=='audid':
            from feats_audid import AudIDset
            self.feats_train = AudIDset(f'{feats_name}_train', df_train)
            self.feats_train.extract()
            self.feats_train.filter()
            self.feats_test = AudIDset(f'{feats_name}_test', df_test)
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
        elif feats_type=='spectra':
            # compute the spectrograms
            test_specs = Spectraloader(f'{feats_name}_test', df_test)
            test_specs.make_feats()
            self.feats_test = test_specs.get_loader()
            train_specs = Spectraloader(f'{feats_name}_train', df_train)
            train_specs.make_feats()
            self.feats_train = train_specs.get_loader()
        else:
            self.util.error(f'unknown feats_type: {feats_type}')
        self.scale()

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
        """Start up the runs."""
        self.runmgr.do_runs()
        # access the results per run
        self.reports = self.runmgr.reports
        self.collect_reports()
        return self.results, self.train_results, self.losses

    def collect_reports(self):
        self.results, self.losses, self.train_results = [], [], []
        for r in self.reports:
            self.results.append(r.get_result().test)
            self.losses.append(r.get_result().loss)
            self.train_results.append(r.get_result().train)
