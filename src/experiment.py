from opensmile.core.define import FeatureSet
from dataset import Dataset
from emodb import Emodb
from opensmileset import Opensmileset
from runmanager import Runmanager
import ast # To convert strings to objects
import pandas as pd

class Experiment:
    """Main class specifying an experiment"""
    
    name = '' # A name for the experiment. Will be used to label result and temporary files.
    datasets = [] # List to store the datasets that are loaded
    df_train = None # The training dataframe
    df_test = None # The evaluation dataframe
    feats_train = None # The training features
    feats_test = None # The test features
    config = None # The configuration object
    runmgr = None  # The manager object for the runs
    labels = None # set of string values for the categories
    values = None # set of numerical values encoding the classes 

    def __init__(self, name, config):
        """Constructor: takes a name and the config object"""
        self.name = name
        self.config = config
    
    def load_datasets(self):
        """Load all databases specified in the configuration and map the labels"""
        ds = ast.literal_eval(self.config['DATA']['databases'])
        for d in ds:
            if d == 'emodb':
                data = Emodb(self.config)
            else:
                data = Dataset(self.config, d)
            data.load()
            data.prepare_labels()
            self.datasets.append(data)

    def fill_train_and_tests(self):
        """Set up train and development sets. The method should be specified in the config."""
        self.df_train, self.df_test = pd.DataFrame(), pd.DataFrame()
        for d in self.datasets:
            d.split()
            self.df_train = self.df_train.append(d.df_train)
            self.df_test = self.df_test.append(d.df_test)
        

    def extract_feats(self):
        """Extract the features for train and dev sets. They will be stpred on disk and need to be removed manually."""
        df_train, df_test = self.df_train, self.df_test
        feats_name = "_".join(ast.literal_eval(self.config['DATA']['databases']))
        self.feats_train = Opensmileset(f'{feats_name}_os_feats_train', self.config, df_train)
        self.feats_train.extract()
        self.feats_test = Opensmileset(f'{feats_name}_os_feats_test', self.config, df_test)
        self.feats_test.extract()

    def init_runmanager(self):
        """Initialize the manager object for the runs."""
        self.runmgr = Runmanager(self.config, self.df_train, self.df_test, self.feats_train, self.feats_test)

    def run(self):
        """Start up the runs."""
        self.runmgr.do_runs()