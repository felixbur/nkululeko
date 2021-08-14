from opensmile.core.define import FeatureSet
from dataset import Dataset
from emodb import Emodb
from opensmileset import Opensmileset
from runmanager import Runmanager
from util import Util
import ast # To convert strings to objects
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Experiment:
    """Main class specifying an experiment"""
    
    name = '' # A name for the experiment. Will be used to label result and temporary files.
    datasets = {} # Dictionary to store the datasets that are loaded
    df_train = None # The training dataframe
    df_test = None # The evaluation dataframe
    feats_train = None # The training features
    feats_test = None # The test features
    config = None # The configuration object
    runmgr = None  # The manager object for the runs
    labels = None # set of string values for the categories
    values = None # set of numerical values encoding the classes 

    def __init__(self, config):
        """Constructor: takes a name and the config object"""
        self.name = config['EXP']['name']
        self.config = config
        self.util = Util(config)

    def load_datasets(self):
        """Load all databases specified in the configuration and map the labels"""
        ds = ast.literal_eval(self.config['DATA']['databases'])
        self.datasets = {}
        for d in ds:
            if d == 'emodb':
                data = Emodb(self.config)
            else:
                data = Dataset(self.config, d)
            data.load()
            data.prepare_labels()
            self.datasets.update({d: data})

    def fill_train_and_tests(self):
        """Set up train and development sets. The method should be specified in the config."""
        self.df_train, self.df_test = pd.DataFrame(), pd.DataFrame()
        strategy = self.config['DATA']['strategy']
        # some datasets against others in their entirety
        if strategy == 'cross_data':
            train_dbs = ast.literal_eval(self.config['DATA']['trains'])
            test_dbs = ast.literal_eval(self.config['DATA']['tests'])
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
            target = self.config['DATA']['target']
            self.label_encoder = LabelEncoder()
            self.df_train[target] = self.label_encoder.fit_transform(self.df_train[target])
            self.df_test[target] = self.label_encoder.transform(self.df_test[target])
        else:
            pass

    def extract_feats(self):
        """Extract the features for train and dev sets. They will be stored on disk and need to be removed manually."""
        df_train, df_test = self.df_train, self.df_test
        feats_name = "_".join(ast.literal_eval(self.config['DATA']['databases']))
        strategy = self.config['DATA']['strategy']
        feats_name = f'{feats_name}_{strategy}_os_feats'
        self.feats_train = Opensmileset(f'{feats_name}_train', self.config, df_train)
        self.feats_train.extract()
        self.feats_train.filter()
        self.feats_test = Opensmileset(f'{feats_name}_test', self.config, df_test)
        self.feats_test.extract()
        self.feats_test.filter()

    def init_runmanager(self):
        """Initialize the manager object for the runs."""
        self.runmgr = Runmanager(self.config, self.df_train, self.df_test, self.feats_train, self.feats_test)

    def run(self):
        """Start up the runs."""
        self.runmgr.do_runs()
        # access the results per run
        self.results = self.runmgr.results
        return self.results[-1]