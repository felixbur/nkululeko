""" test_predictor.py
    Predict targets from a model and save as csv file.

"""
import nkululeko.glob_conf as glob_conf
from nkululeko.utils.util import Util
import pandas as pd
from nkululeko.data.dataset import Dataset
from nkululeko.feature_extractor import FeatureExtractor
from nkululeko.scaler import Scaler
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Test_predictor:
    def __init__(self, model, orig_df, labenc, name):
        """Constructor setting up name and configuration"""
        self.model = model
        self.orig_df = orig_df
        self.label_encoder = labenc
        self.target = glob_conf.config["DATA"]["target"]
        self.util = Util("test_predictor")
        self.name = name

    def predict_and_store(self):
        label_data = self.util.config_val("DATA", "label_data", False)
        if label_data:
            data = Dataset(label_data)
            data.load()
            data.prepare_labels()
            data_df = self.util.make_segmented_index(data.df)
            data_df.is_labeled = data.is_labeled

            featextractor = FeatureExtractor(data_df, label_data, "")
            feats_df = featextractor.extract()
            scale = self.util.config_val("FEATS", "scale", False)
            labelenc = LabelEncoder()
            data_df[self.target] = labelenc.fit_transform(data_df[self.target])
            #            data_df[self.target] = self.label_encoder.fit_transform(data_df[self.target])
            if scale:
                self.scaler = Scaler(data_df, None, feats_df, None, scale)
                feats_df, _ = self.scaler.scale()
            self.model.set_testdata(data_df, feats_df)
            predictions = self.model.get_predictions()
            df = pd.DataFrame(index=data_df.index)
            df["speaker"] = data_df["speaker"]
            df["gender"] = data_df["gender"]
            df[self.target] = labelenc.inverse_transform(predictions.tolist())
            df.to_csv(self.name)
        else:
            predictions = self.model.get_predictions()
            # print(predictions)
            df = pd.DataFrame(index=self.orig_df.index)
            df["speaker"] = self.orig_df["speaker"]
            df["gender"] = self.orig_df["gender"]
            df[self.target] = self.label_encoder.inverse_transform(predictions)
            # df[self.target] = predictions
            df.to_csv(self.name)
