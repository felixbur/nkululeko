import glob_conf
from util import Util
import pandas as pd

class Test_predictor():


    def __init__(self, model, orig_df, label_encoder, name):
        """Constructor setting up name and configuration"""
        self.model = model
        self.orig_df = orig_df
        self.label_encoder = label_encoder
        self.target = glob_conf.config['DATA']['target']
        self.util = Util()
        self.name = name

    def predict_and_store(self):
        predictions = self.model.get_predictions()
        df = pd.DataFrame(index = self.orig_df.index)
        df['speaker'] = self.orig_df['speaker']
        df['gender'] = self.orig_df['gender']
        df[self.target] = self.label_encoder.inverse_transform(predictions)
        df.to_csv(self.name)