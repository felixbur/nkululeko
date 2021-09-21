# dataset.py
import audformat
import pandas as pd
import ast
import os
from random import sample
from util import Util
import glob_conf

class Dataset:
    """ Class to represent datasets"""
    name = '' # An identifier for the dataset
    config = None # The configuration 
    db = None # The database object
    df = None # The whole dataframe
    df_train = None # The training split
    df_test = None # The evaluation split

    def __init__(self, name):
        """Constructor setting up name and configuration"""
        self.name = name
        self.target = glob_conf.config['DATA']['target']
        self.util = Util()

    def load(self):
        """Load the dataframe with files, speakers and task labels"""
        self.util.debug(f'loading {self.name}')
        root = glob_conf.config['DATA'][self.name]
        db = audformat.Database.load(root)
        # map the audio file paths 
        db.map_files(lambda x: os.path.join(root, x))
        # the dataframe with all other information 
        df_files = self.util.config_val('DATA', f'{self.name}.files_table', 'files')
        try :
            df = db[df_files].df
        except audformat.core.errors.BadKeyError:
            # if no such table exists, create a new one and hope for the best
            df = pd.DataFrame()
        try :
           # There might be a separate table with the targets, e.g. emotion or age    
            df_target = db.tables[self.target].df
            df[self.target] = df_target[self.target]
        except KeyError:
            pass
        try: 
            # for experiments that do separate sex models
            s = glob_conf.config['DATA']['sex']
            df = df[df.gender==s]
        except KeyError:
            pass 
        try:
            df['gender'] = db['files']['speaker'].get(map='gender')
        except (ValueError, audformat.errors.BadKeyError) as e:
            pass
        try:
            df[self.target] = db['files']['speaker'].get(map=self.target)
        except (ValueError, audformat.core.errors.BadKeyError) as e:
            pass
        self.df = df
        self.db = db

    def split(self):
        """Split the datbase into train and development set"""
        store = self.util.get_path('store')
        storage_test = f'{store}{self.name}_testdf.pkl'
        storage_train = f'{store}{self.name}_traindf.pkl'
        split_strategy = self.util.config_val('DATA', self.name+'.split_strategy', 'database')
        # 'database' (default), 'speaker_split', 'specified', 'reuse'
        if split_strategy == 'database':
            #  use the splits from the database
            testdf = self.db.tables[self.target+'.test'].df
            traindf = self.db.tables[self.target+'.train'].df
            # use only the train and test samples that were not perhaps filtered out by an earlier processing step
            self.df_test = self.df.loc[self.df.index.intersection(testdf.index)]
            self.df_train = self.df.loc[self.df.index.intersection(traindf.index)]
        elif split_strategy == 'specified':
            traindf, testdf = pd.DataFrame(), pd.DataFrame()
            # try to load some dataframes for testing
            try:
                test_tables =  ast.literal_eval(glob_conf.config['DATA'][self.name+'.test_tables'])
                for test_table in test_tables:
                    testdf = testdf.append(self.db.tables[test_table].df)
            except KeyError:
                pass
            try:
                train_tables = ast.literal_eval(glob_conf.config['DATA'][self.name+'.train_tables'])
                for train_table in train_tables:
                    traindf = traindf.append(self.db.tables[train_table].df)
            except KeyError:
                pass
            # use only the train and test samples that were not perhaps filtered out by an earlier processing step
            self.df_test = self.df.loc[self.df.index.intersection(testdf.index)]
            self.df_train = self.df.loc[self.df.index.intersection(traindf.index)]
            # it might be necessary to copy the target values 
            try:
                self.df_test[self.target] = testdf[self.target]
            except KeyError:
                pass # if the dataframe is empty
            try:
                self.df_train[self.target] = traindf[self.target]
            except KeyError:
                pass # if the dataframe is empty
        elif split_strategy == 'speaker_split':
            self.split_speakers()
        elif split_strategy == 'reuse':
            self.df_test = pd.read_pickle(storage_test)
            self.df_train = pd.read_pickle(storage_train)
        """Bin target values if they are continous but a classification experiment should be done"""
        self.check_continous_classification(self.df_train)
        self.check_continous_classification(self.df_test)
        # remember the target in case they get labelencoded later
        self.df_test['class_label'] = self.df_test[self.target]
        self.df_train['class_label'] = self.df_train[self.target]
        # remember the splits for future use
        self.df_test.to_pickle(storage_test)
        self.df_train.to_pickle(storage_train)
        if self.util.config_val('PLOT', 'value_counts', False):
            self.plot_distribution()

    def plot_distribution(self):
        from plots import Plots
        plot = Plots()
        all_df = self.df_test.append(self.df_train)
        plot.describe_df(self.name, all_df, self.target, f'{self.name}_distplot.png')
        if self.df_test.shape[0]>0:
            plot.describe_df(self.name+' dev', self.df_test, self.target, f'{self.name}_test_distplot.png')
        if self.df_train.shape[0]>0:
            plot.describe_df(self.name+' train', self.df_train, self.target, f'{self.name}_train_distplot.png')

    def split_speakers(self):
        """One way to split train and eval sets: Specify percentage of evaluation speakers"""
        test_percent = int(self.util.config_val('DATA', self.name+'.testsplit', 50))
        df = self.df
        s_num = df.speaker.nunique()
        test_num = int(s_num * (test_percent/100))        
        test_spkrs =  sample(list(df.speaker.unique()), test_num)
        self.df_test = df[df.speaker.isin(test_spkrs)]
        self.df_train = df[~df.index.isin(self.df_test.index)]
        self.util.debug(f'{self.name}: [{self.df_train.shape[0]}/{self.df_test.shape[0]}] samples in train/test')
        # because this generates new train/test sample quantaties, the feature extraction has to be done again
        glob_conf.config['DATA']['needs_feature_extraction'] = 'true'

    def prepare_labels(self):
        """Bin target values if they are continous but a classification experiment sould be done"""
        self.check_continous_classification(self.df)
        """Rename the labels and remove the ones that are not needed."""
        try :
            mapping = ast.literal_eval(glob_conf.config['DATA'][f'{self.name}.mapping'])
            target = glob_conf.config['DATA']['target']
            labels = ast.literal_eval(glob_conf.config['DATA']['labels'])
            df = self.df
            df[target] = df[target].map(mapping)
            self.df = df[df[target].isin(labels)]
            self.util.debug(f'for dataset {self.name} mapped {mapping}')
            self.util.debug(f'Categories: {self.df[target].unique()}')

        except KeyError:
            pass

    def check_continous_classification(self, df):
        datatype = self.util.config_val('DATA', 'type', 'dummy')
        if self.util.exp_is_classification() and datatype == 'continuous':
            self.util.debug('binning continuous variable to categories')
            cat_vals = self.util.continuous_to_categorical(df[self.target])
            df[self.target] = cat_vals
 