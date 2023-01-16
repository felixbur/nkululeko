# scaler.py


from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
from nkululeko.util import Util

class Scaler:
    """
        class to normalize speech parameters   
    """
    
    def __init__(self, train_data_df, test_data_df, train_feats, test_feats, scaler_type):
        '''
        Initializer.

                Parameters:
                        train_data_df (pd.DataFrame): The training dataframe with speakers.
                            only needed for speaker normalization
                        test_data_df (pd.DataFrame): The test dataframe with speakers
                            only needed for speaker normalization
                        train_feats (pd.DataFrame): The train features dataframe 
                        test_feats (pd.DataFrame): The test features dataframe (can be None) 
        '''
        self.util = Util()
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'speaker':
            self.scaler = StandardScaler()
        else:
            self.util.error('unknown scaler: '+scaler_type)
        self.scaler_type = scaler_type
        self.feats_train = train_feats
        self.data_train = train_data_df
        self.feats_test = test_feats
        self.data_test = test_data_df

    def scale(self):
        '''
        Actually scales/normalizes.

                Returns:
                        train_feats (pd.DataFrame): The scaled train features dataframe 
                        test_feats (pd.DataFrame): The scaled test features dataframe (can be None) 
        '''
        if self.scaler_type != 'speaker':
            self.util.debug('scaling features based on training set')
            return self.scale_all()
        else:
            self.util.debug('scaling features per speaker based on training')
            return self.speaker_scale()

    def scale_all(self):
        self.scaler.fit(self.feats_train.values)
        self.feats_train = self.scale_df(self.feats_train)
        if self.feats_test is not None:
            self.feats_test = self.scale_df(self.feats_test)
        return self.feats_train, self.feats_test

    def scale_df(self, df):
        scaled_features = self.scaler.fit_transform(df.values)
        df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
        return df 

    def speaker_scale(self):
        self.feats_train = self.speaker_scale_df(self.data_train, self.feats_train)
        if self.feats_test is not None:
            self.feats_test = self.speaker_scale_df(self.data_test, self.feats_test)
        return [self.feats_train, self.feats_test]


    def speaker_scale_df(self, df, feats_df):
        for speaker in df['speaker'].unique():
            indices = df.loc[df['speaker'] == speaker].index
            feats_df.loc[indices, :] = self.scaler.fit_transform(feats_df.loc[indices, :]) 
        return feats_df