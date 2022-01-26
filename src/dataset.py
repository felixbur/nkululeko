# dataset.py
import audformat
import pandas as pd
import ast
import os
from random import sample
from util import Util
from plots import Plots
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
        self.plot = Plots()
        self.limit = int(self.util.config_val('DATA', f'{self.name}.limit', 0))

    def load(self):
        """Load the dataframe with files, speakers and task labels"""
        self.util.debug(f'loading {self.name}')
        root = glob_conf.config['DATA'][self.name]
        db = audformat.Database.load(root)
        # map the audio file paths 
        db.map_files(lambda x: os.path.join(root, x))
        # the dataframes (potentially more than one) with at least the file names
        df_files = self.util.config_val('DATA', f'{self.name}.files_tables', '[\'files\']')
        df_files_tables =  ast.literal_eval(df_files)
        # The label for the target column 
        self.col_label = self.util.config_val('DATA', f'{self.name}.label', self.target)
        df, got_target, got_speaker, got_gender = self.get_df_for_lists(db, df_files_tables)
        if False in {got_target, got_speaker, got_gender}:
            try :
            # There might be a separate table with the targets, e.g. emotion or age    
                df_targets = self.util.config_val('DATA', f'{self.name}.target_tables', f'[\'{self.target}\']')
                df_target_tables =  ast.literal_eval(df_targets)
                df_target, got_target2, got_speaker2, got_gender2 = self.get_df_for_lists(db, df_target_tables)
                got_target = got_target2 or got_target
                got_speaker = got_speaker2 or got_speaker
                got_gender = got_gender2 or got_gender
                if got_target2:
                    df[self.target] = df_target[self.target]
                if got_speaker2:
                    df['speaker'] = df_target['speaker']
                if got_gender2:
                    df['gender'] = df_target['gender']
            except audformat.core.errors.BadKeyError:
                pass
        try: 
            # for experiments that do separate sex models
            s = glob_conf.config['DATA']['sex']
            df = df[df.gender==s]
        except KeyError:
            pass 
        # remember the target in case they get labelencoded later
        df['class_label'] = df[self.target]
        self.df = df
        self.db = db
        self.util.debug(f'Loaded database {self.name} with {df.shape[0]} '\
            f'samples: got targets: {got_target}, got speakers: {got_speaker}, '\
            f'got sexes: {got_gender}')
        if self.util.config_val('DATA', f'{self.name}.value_counts', False):
            if not got_gender or not got_speaker:
                self.util.error('can\'t plot value counts if no speaker or gender is given')
            else:
                self.plot.describe_df(self.name, df, self.target, f'{self.name}_distplot.png')

    def get_df_for_lists(self, db, df_files):
        got_target, got_speaker, got_gender = False, False, False
        df = pd.DataFrame()
        for table in df_files:
            if self.limit:
                source_df = db.tables[table].df.iloc[:self.limit]
            else:
                source_df = db.tables[table].df
            # create a dataframe with the index (the filenames)
            df_local = pd.DataFrame(index=source_df.index)
            # try to get the targets from this dataframe
            try:
                # try to get the target values
                df_local[self.target] = source_df[self.col_label]
                got_target = True
            except (KeyError, ValueError, audformat.errors.BadKeyError) as e:
                pass
            try:
                # try to get the speaker values
                df_local['speaker'] = source_df['speaker']
                got_speaker = True
            except (KeyError, ValueError, audformat.errors.BadKeyError) as e:
                pass
            try:
                # try to get the gender values
                df_local['gender'] = source_df['gender']
                got_gender = True
            except (KeyError, ValueError, audformat.errors.BadKeyError) as e:
                pass
            try:
                # also it might be possible that the sex is part of the speaker description
                if self.limit:
                    df_local['gender'] = db[table]['speaker'].get(map='gender').iloc[:self.limit]
                else:
                    df_local['gender'] = db[table]['speaker'].get(map='gender')

                got_gender = True
            except (ValueError, audformat.errors.BadKeyError) as e:
                pass
            try:
                # same for the target, e.g. "age"
                if self.limit:
                    df_local[self.target] = db[table]['speaker'].get(map=self.target).iloc[:self.limit]
                else:
                    df_local[self.target] = db[table]['speaker'].get(map=self.target)
                got_target = True
            except (ValueError, audformat.core.errors.BadKeyError) as e:
                pass
            df = df.append(df_local)
        return df, got_target, got_speaker, got_gender

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
        elif split_strategy == 'train':
            self.df_train = self.df
            self.df_test = pd.DataFrame()
        elif split_strategy == 'test':
            self.df_test = self.df
            self.df_train = pd.DataFrame()
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

        if self.df_test.shape[0]>0:
            self.finish_up(self.df_test, 'test', storage_test)
        if self.df_train.shape[0]>0:
            self.finish_up(self.df_train, 'train', storage_train)

    def finish_up(self, df, name, storage):
        # Bin target values if they are continous but a classification experiment should be done
        # self.check_continous_classification(df)
        # remember the splits for future use
        df.to_pickle(storage)
        if self.util.config_val('DATA', f'{self.name}.value_counts', False):
            all_df = self.df_test.append(self.df_train)
            self.plot.describe_df(self.name, all_df, self.target, f'{self.name}_distplot.png')
            self.plot.describe_df(self.name+' dev', self.df_test, self.target, f'{self.name}_{name}_distplot.png')

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
        glob_conf.config['FEATS']['needs_feature_extraction'] = 'True'

    def prepare_labels(self):
        strategy = self.util.config_val('DATA', 'strategy', 'train_test')
        if strategy == 'cross_data':
            self.df = self.map_labels(self.df)
            # Bin target values if they are continuous but a classification experiment should be done
            self.map_continuous_classification(self.df)
        elif strategy == 'train_test':        
            self.df_train = self.map_labels(self.df_train)
            self.df_test = self.map_labels(self.df_test)
            self.map_continuous_classification(self.df_train)
            self.map_continuous_classification(self.df_test)

    def map_labels(self, df):
        if df.shape[0]==0 or not self.util.exp_is_classification() \
            or self.check_continuous_classification() :
            return df
        """Rename the labels and remove the ones that are not needed."""
        target = glob_conf.config['DATA']['target']
        try :
            # see if a special mapping should be used
            mapping = ast.literal_eval(glob_conf.config['DATA'][f'{self.name}.mapping'])
            df[target] = df[target].map(mapping)
            self.util.debug(f'for dataset {self.name} mapped {mapping}')
        except KeyError:
            pass
        # remove labels that are not in the labels list
        try :
            labels = ast.literal_eval(glob_conf.config['DATA']['labels'])
            df = df[df[target].isin(labels)]
            # self.util.debug(f'Categories: {df[target].unique()}')
        except KeyError:
            pass
        return df

    def check_continuous_classification(self):
        datatype = self.util.config_val('DATA', 'type', 'dummy')
        if self.util.exp_is_classification() and datatype == 'continuous':
            return True
        return False

    def map_continuous_classification(self, df):
        """Map labels to bins for continuous data that should be classified"""
        if self.check_continuous_classification():
            self.util.debug('binning continuous variable to categories')
            cat_vals = self.util.continuous_to_categorical(df[self.target])
            df[self.target] = cat_vals
            labels = ast.literal_eval(glob_conf.config['DATA']['labels'])
            df['class_label'] = df[self.target]
#            print(df['class_label'].unique())
            for i, l in enumerate(labels):
                df['class_label'] = df['class_label'].replace(i, str(l))