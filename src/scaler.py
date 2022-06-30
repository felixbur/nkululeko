# scaler.py


from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
from util import Util
import glob_conf

class Scaler:
    # class to normalize speech parameters

    def __init__(self, train_data_df, test_data_df, train_feats, test_feats, scaler_type):
        self.util = Util()
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'speaker':
            pass
        else:
            self.util.error('unknown scaler: '+scaler_type)
        self.scaler_type = scaler_type
        self.feats_train = train_feats
        self.data_train = train_data_df
        self.feats_test = test_feats
        self.data_test = test_data_df

    def scale(self):
        if self.scaler_type != 'speaker':
            self.util.debug('scaling features based on training')
            return self.scale_all()
        else:
            self.util.debug('scaling features per speaker based on training')
            return self.speaker_scale()

    def scale_all(self):
        self.scaler.fit(self.feats_train.values)
        self.feats_train = self.scale_df(self.feats_train)
        self.feats_test = self.scale_df(self.feats_test)
        return self.feats_train, self.feats_test

    def scale_df(self, df):
        scaled_features = self.scaler.fit_transform(df.values)
        df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
        return df 

    def speaker_scale(self):
        self.feats_train = self.speaker_scale_df(self.data_train, self.feats_train)
        self.feats_test = self.speaker_scale_df(self.data_test, self.feats_test)
        return self.feats_train, self.feats_test


    def speaker_scale_df(self, df, feats_df):
        for speaker in df['speaker'].unique():
            indices = df.loc[df['speaker'] == speaker].index
            feats_df.loc[indices, :] = self.scaler.fit_transform(feats_df.loc[indices, :]) 
        return feats_df