"""test_predictor.py.

    Predict targets from a model and save as csv file.

"""

import ast

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import nkululeko.glob_conf as glob_conf
from nkululeko.data.dataset import Dataset
from nkululeko.feature_extractor import FeatureExtractor
from nkululeko.scaler import Scaler
from nkululeko.utils.util import Util


class TestPredictor:
    def __init__(self, model, orig_df, labenc, name):
        """Constructor setting up name and configuration."""
        self.model = model
        self.orig_df = orig_df
        self.label_encoder = labenc
        self.target = glob_conf.config["DATA"]["target"]
        self.util = Util("test_predictor")
        self.name = name

    def predict_and_store(self):
        label_data = self.util.config_val("DATA", "label_data", False)
        result = 0
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
            test_dbs = ast.literal_eval(glob_conf.config["DATA"]["tests"])
            test_dbs_string = "_".join(test_dbs)
            predictions, _ = self.model.get_predictions()
            report = self.model.predict()
            result = report.result.get_result()
            report.set_filename_add(f"test-{test_dbs_string}")
            self.util.print_best_results([report])
            report.plot_confmatrix(self.util.get_plot_name(), 0)
            report.print_results(0)
            df = self.orig_df.copy()
            df["predictions"] = self.label_encoder.inverse_transform(predictions)
            target = self.util.config_val("DATA", "target", "emotion")
            if "class_label" in df.columns:
                df = df.drop(columns=[target])
                df = df.rename(columns={"class_label": target})
            df.to_csv(self.name)
        self.util.debug(f"results stored in {self.name}")
        return result
