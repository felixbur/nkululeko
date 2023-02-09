# dataset.py
import audformat
#import audb
import pandas as pd
import ast
import os
from random import sample
from nkululeko.util import Util
from nkululeko.plots import Plots
import nkululeko.glob_conf as glob_conf
import os.path
from audformat.utils import duration
import nkululeko.filter_data as filter

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
        self.limit = int(self.util.config_val_data(self.name, 'limit', 0))
        self.start_fresh = eval(self.util.config_val('DATA', 'no_reuse', 'False'))
        self.is_labeled, self.got_speaker, self.got_gender = False, False, False 


    def _get_tables(self):
        tables = []
        targets = self.util.config_val_data(self.name, 'target_tables', False)
        if targets:
            target_tables = ast.literal_eval(targets)
            tables += target_tables
        files = self.util.config_val_data(self.name, 'files_tables', False)
        if files:
            files_tables = ast.literal_eval(files)
            tables += files_tables
        tests = self.util.config_val_data(self.name, 'test_tables', False)
        if tests:
            test_tables = ast.literal_eval(tests)
            tables += test_tables
        trains = self.util.config_val_data(self.name, 'train_tables', False)
        if trains:
            train_tables = ast.literal_eval(trains)
            tables += train_tables
        return tables


    def load(self):
        """Load the dataframe with files, speakers and task labels"""
        # store the dataframe
        store = self.util.get_path('store')
        store_file = f'{store}{self.name}.pkl'
        self.util.debug(f'{self.name}: loading ...')
        self.got_speaker, self.got_gender = False, False 
        if not self.start_fresh and os.path.isfile(store_file):
            self.util.debug(f'{self.name}: reusing previously stored file {store_file}')
            self.df = pd.read_pickle(store_file)
            self.is_labeled = self.target in self.df
            self.got_gender = 'gender' in self.df
            self.got_speaker = 'speaker' in self.df
            self.util.debug(f'{self.name}: loaded with {self.df.shape[0]} '\
                f'samples: got targets: {self.is_labeled}, got speakers: {self.got_speaker}, '\
                f'got sexes: {self.got_gender}')        
            return
        root = self.util.config_val_data(self.name, '', '')
        self.util.debug(f'{self.name}: loading from {root}')
        try:
            db = audformat.Database.load(root)
        except FileNotFoundError:
            self.util.error( f'{self.name}: no database found at {root}')
        tables = self._get_tables()
        self.util.debug(f'{self.name}: loading tables: {tables}')
        #db = audb.load(root, )
        # map the audio file paths 
        db.map_files(lambda x: os.path.join(root, x))
        # the dataframes (potentially more than one) with at least the file names
        df_files = self.util.config_val_data(self.name, 'files_tables', '[\'files\']')
        df_files_tables =  ast.literal_eval(df_files)
        # The label for the target column 
        self.col_label = self.util.config_val_data(self.name, 'label', self.target)
        df, self.is_labeled, self.got_speaker, self.got_gender = self._get_df_for_lists(db, df_files_tables)
        if False in {self.is_labeled, self.got_speaker, self.got_gender}:
            try :
            # There might be a separate table with the targets, e.g. emotion or age    
                df_targets = self.util.config_val_data(self.name, 'target_tables', f'[\'{self.target}\']')
                df_target_tables =  ast.literal_eval(df_targets)
                df_target, got_target2, got_speaker2, got_gender2 = self._get_df_for_lists(db, df_target_tables)
                self.is_labeled = got_target2 or self.is_labeled
                self.got_speaker = got_speaker2 or self.got_speaker
                self.got_gender = got_gender2 or self.got_gender
                if got_target2:
                    df[self.target] = df_target[self.target]
                if got_speaker2:
                    df['speaker'] = df_target['speaker']
                if got_gender2:
                    df['gender'] = df_target['gender']
            except audformat.core.errors.BadKeyError:
                pass

        if self.is_labeled:
            # remember the target in case they get labelencoded later
            df['class_label'] = df[self.target]
        df.is_labeled = self.is_labeled
        df.got_gender = self.got_gender
        df.got_speaker = self.got_speaker
        self.df = df
        self.db = db
        self.df.is_labeled = self.is_labeled
        print('huhu')

    def prepare(self):
        # Perform some filtering if desired
        required = self.util.config_val_data(self.name, 'required', False)
        if required:
            pre = self.df.shape[0]
            self.df = self.df[self.df[required].notna()]
            post = self.df.shape[0]
            self.util.debug(f'{self.name}: kept {post} samples with {required} (from {pre}, filtered {pre-post})')
        samples_per_speaker = self.util.config_val_data(self.name, 'max_samples_per_speaker', False)
        if samples_per_speaker:
            pre = self.df.shape[0]
            self.df = filter.limit_speakers(self.df, int(samples_per_speaker))            
            post = self.df.shape[0]
            self.util.debug(f'{self.name}: kept {post} samples with {samples_per_speaker} per speaker (from {pre}, filtered {pre-post})')
        if self.limit:
            pre = self.df.shape[0]
            self.df = self.df.sample(self.limit)
            post = self.df.shape[0]
            self.util.debug(f'{self.name}: limited to {post} samples (from {pre}, filtered {pre-post})')
        sex = self.util.config_val('DATA', 'sex', False)
        if sex:
            # for experiments that do separate sex models
            pre = self.df.shape[0]
            self.df = self.df[self.df.gender==sex]
            post = self.df.shape[0]
            self.util.debug(f'{self.name}: limited to {post} samples with sex {sex} (from {pre}, filtered {pre-post})')
            self.df.is_labeled = self.is_labeled
            self.df.got_gender = self.got_gender
            self.df.got_speaker = self.got_speaker
        min_dur = self.util.config_val_data(self.name, 'min_duration_of_sample', False)
        if min_dur:
            pre = self.df.shape[0]
            self.df = filter.filter_min_dur(self.df, min_dur)
            post = self.df.shape[0]
            self.util.debug(f'{self.name}: dropped {pre-post} shorter than {min_dur} seconds (from {pre} to {post} samples)')

        max_dur = self.util.config_val_data(self.name, 'max_duration_of_sample', False)
        if max_dur:
            pre = self.df.shape[0]
            self.df = filter.filter_max_dur(self.df, max_dur)
            post = self.df.shape[0]
            self.util.debug(f'{self.name}: dropped {pre-post} longer than {max_dur} seconds (from {pre} to {post} samples)')

        self.util.debug(f'{self.name}: loaded data with {self.df.shape[0]} '\
            f'samples: got targets: {self.is_labeled}, got speakers: {self.got_speaker}, '\
            f'got sexes: {self.got_gender}')

        if self.got_speaker and self.util.config_val_data(self.name, 'rename_speakers', False):
            # we might need to append the database name to all speakers in case other datbaases have the same speaker names
            self.df.speaker = self.df.speaker.apply(lambda x: self.name+x)

        # store the dataframe
        store = self.util.get_path('store')
        store_file = f'{store}{self.name}.pkl'
        self.df.to_pickle(store_file)


    def _get_df_for_lists(self, db, df_files):
        is_labeled, got_speaker, got_gender = False, False, False
        df = pd.DataFrame()
        for table in df_files:
            source_df = db.tables[table].df
            # create a dataframe with the index (the filenames)
            df_local = pd.DataFrame(index=source_df.index)
            # try to get the targets from this dataframe
            try:
                # try to get the target values
                df_local[self.target] = source_df[self.col_label]
                is_labeled = True
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
                df_local['gender'] = db[table]['speaker'].get(map='gender')

                got_gender = True
            except (ValueError, audformat.errors.BadKeyError) as e:
                pass
            try:
                # same for the target, e.g. "age"
                df_local[self.target] = db[table]['speaker'].get(map=self.target)
                is_labeled = True
            except (ValueError, audformat.core.errors.BadKeyError) as e:
                pass
            df = pd.concat([df, df_local])
        return df, is_labeled, got_speaker, got_gender


    def split(self):
        """Split the datbase into train and development set"""
        store = self.util.get_path('store')
        storage_test = f'{store}{self.name}_testdf.pkl'
        storage_train = f'{store}{self.name}_traindf.pkl'
        split_strategy = self.util.config_val_data(self.name,'split_strategy', 'database')
        self.util.debug(f'splitting database {self.name} with strategy {split_strategy}')
        # 'database' (default), 'speaker_split', 'specified', 'reuse'
        if split_strategy != 'speaker_split' and not self.start_fresh:
            # check if the splits have been computed previously (not for speaker split)
            if os.path.isfile(storage_train) and os.path.isfile(storage_test):
                # if self.util.config_val_data(self.name, 'test_tables', False):
                self.util.debug(f'splits: reusing previously stored test file {storage_test}')
                self.df_test = pd.read_pickle(storage_test)
                self.util.debug(f'splits: reusing previously stored train file {storage_train}')
                self.df_train = pd.read_pickle(storage_train)
                return
            elif os.path.isfile(storage_train):
                self.util.debug(f'splits: reusing previously stored train file {storage_train}')
                self.df_train = pd.read_pickle(storage_train)
                self.df_test = pd.DataFrame()
                return
            elif os.path.isfile(storage_test):
                self.util.debug(f'splits: reusing previously stored test file {storage_test}')
                self.df_test = pd.read_pickle(storage_test)
                self.df_train = pd.DataFrame()
                return
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
            entry_test_tables = self.util.config_val_data(self.name, 'test_tables', False)
            if entry_test_tables: 
                test_tables =  ast.literal_eval(entry_test_tables)
                for test_table in test_tables:
                    testdf = testdf.append(self.db.tables[test_table].df)
            entry_train_tables = self.util.config_val_data(self.name, 'train_tables', False)
            if entry_train_tables: 
                train_tables =  ast.literal_eval(entry_train_tables)
                for train_table in train_tables:
                    traindf = pd.concat([traindf, self.db.tables[train_table].df])
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
        elif split_strategy == 'random':
            self.random_split()
        elif split_strategy == 'reuse':
            self.util.debug(f'{self.name}: trying to reuse data splits')
            self.df_test = pd.read_pickle(storage_test)
            self.df_train = pd.read_pickle(storage_train)

        if self.df_test.shape[0]>0:
            self.df_test = self.finish_up(self.df_test, storage_test)
        if self.df_train.shape[0]>0:
            self.df_train = self.finish_up(self.df_train, storage_train)
        
    def finish_up(self, df, storage):
        # Bin target values if they are continuous but a classification experiment should be done
        # self.check_continuous_classification(df)
        # remember the splits for future use
        df.is_labeled = self.is_labeled
        self.df_test.is_labeled = self.is_labeled
        df.to_pickle(storage)
        return df

    def split_speakers(self):
        """One way to split train and eval sets: Specify percentage of evaluation speakers"""
        test_percent = int(self.util.config_val_data(self.name, 'testsplit', 50))
        df = self.df
        s_num = df.speaker.nunique()
        test_num = int(s_num * (test_percent/100))        
        test_spkrs =  sample(list(df.speaker.unique()), test_num)
        self.df_test = df[df.speaker.isin(test_spkrs)]
        self.df_train = df[~df.index.isin(self.df_test.index)]
        self.util.debug(f'{self.name}: [{self.df_train.shape[0]}/{self.df_test.shape[0]}] samples in train/test')
        # because this generates new train/test sample quantaties, the feature extraction has to be done again
        glob_conf.config['FEATS']['needs_feature_extraction'] = 'True'

    def random_split(self):
        """One way to split train and eval sets: Specify percentage of random samples"""
        test_percent = int(self.util.config_val_data(self.name, 'testsplit', 50))
        df = self.df
        s_num = len(df)
        test_num = int(s_num * (test_percent/100))        
        test_smpls =  sample(list(df.index), test_num)
        self.df_test = df[df.index.isin(test_smpls)]
        self.df_train = df[~df.index.isin(self.df_test.index)]
        self.util.debug(f'{self.name}: [{self.df_train.shape[0]}/{self.df_test.shape[0]}] samples in train/test')
        # because this generates new train/test sample quantaties, the feature extraction has to be done again
        glob_conf.config['FEATS']['needs_feature_extraction'] = 'True'

    def _add_labels(self, df):
        df.is_labeled = self.is_labeled
        df.got_gender = self.got_gender
        df.got_speaker = self.got_speaker
        return df

    def prepare_labels(self):
        strategy = self.util.config_val('DATA', 'strategy', 'train_test')
        if strategy == 'cross_data':
            self.df = self.map_labels(self.df)
            # Bin target values if they are continuous but a classification experiment should be done
            self.map_continuous_classification(self.df)
            self.df = self._add_labels(self.df)
            if self.util.config_val_data(self.name, 'value_counts', False):
                if not self.got_gender or not self.got_speaker:
                    self.util.error('can\'t plot value counts if no speaker or gender is given')
                else:
                    self.plot.describe_df(self.name, self.df, self.target, f'{self.name}_distplot.png')
        elif strategy == 'train_test':        
            self.df_train = self.map_labels(self.df_train)
            self.df_test = self.map_labels(self.df_test)
            self.map_continuous_classification(self.df_train)
            self.map_continuous_classification(self.df_test)
            self.df_train = self._add_labels(self.df_train)
            self.df_test = self._add_labels(self.df_test)
            if self.util.config_val_data(self.name, 'value_counts', False):
                if not self.got_gender or not self.got_speaker:
                    self.util.error('can\'t plot value counts if no speaker or gender is given')
                else:
                    self.plot.describe_df(self.name, self.df_train, self.target, f'{self.name}_train_distplot.png')
                    self.plot.describe_df(self.name, self.df_test, self.target, f'{self.name}_test_distplot.png')


    def map_labels(self, df):
        pd.options.mode.chained_assignment = None
        if df.shape[0]==0 or not self.util.exp_is_classification() \
            or self.check_continuous_classification():
            return df
        """Rename the labels and remove the ones that are not needed."""
        target = glob_conf.config['DATA']['target']
        # see if a special mapping should be used
        mappings = self.util.config_val_data(self.name, 'mapping', False)
        if mappings:        
            mapping = ast.literal_eval(mappings)
            df[target] = df[target].map(mapping)
            self.util.debug(f'{self.name}: mapped {mapping}')
        # remove labels that are not in the labels list
        labels = ast.literal_eval(glob_conf.config['DATA']['labels'])
        df = df[df[target].isin(labels)]
        # try:
        # except KeyError:
        #     pass
        # remember in case they get encoded later
        df['class_label'] = df[target]
        return df

    def check_continuous_classification(self):
        datatype = self.util.config_val('DATA', 'type', 'dummy')
        if self.util.exp_is_classification() and datatype == 'continuous':
            return True
        return False

    def map_continuous_classification(self, df):
        """Map labels to bins for continuous data that should be classified"""
        if self.check_continuous_classification():
            self.util.debug('{self.name}: binning continuous variable to categories')
            cat_vals = self.util.continuous_to_categorical(df[self.target])
            df[self.target] = cat_vals
            labels = ast.literal_eval(glob_conf.config['DATA']['labels'])
            df['class_label'] = df[self.target]
#            print(df['class_label'].unique())
            for i, l in enumerate(labels):
                df['class_label'] = df['class_label'].replace(i, str(l))