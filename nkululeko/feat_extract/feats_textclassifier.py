import os
import ast
import pandas as pd
from tqdm import tqdm
import transformers
import torch

from nkululeko.feat_extract.featureset import Featureset
import nkululeko.glob_conf as glob_conf


class TextClassifier(Featureset):
    """Class to zero-shot classify text using a pretrained text classification model"""

    def __init__(self, name, data_df, feats_type=None):
        """Constructor.

        Args:
            name (str): Name of the feature set
            data_df (pd.DataFrame): DataFrame containing text data
            feats_type (str, optional): Type of features to extract. Defaults to None.
        """
        super().__init__(name, data_df, feats_type)
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        self.model_initialized = False

    def init_model(self):
        # load model
        modelname = "joeddav/xlm-roberta-large-xnli"
        self.util.debug(f"loading {modelname} model...")

        device_arg = 0 if self.device == "cuda" else -1
        self.model = transformers.pipeline(
            "zero-shot-classification", model="joeddav/xlm-roberta-large-xnli", device=device_arg
        )
        print(f"initialized {modelname} model on {self.device}")
        self.candidate_labels = self.util.config_val(
            "PREDICT", "textclassifier.candidates", False
        )
        if not self.candidate_labels:
            self.util.error(
                "no candidate labels specified for text classification, please set PREDICT/textclassifier.candidates in the config"
            )
        else:
            self.candidate_labels = ast.literal_eval(self.candidate_labels)
            self.util.debug(f"using candidate labels: {self.candidate_labels}")
        self.model_initialized = True

    def extract(self):
        """Extract the features or load them from disk if present."""
        store = self.util.get_path("store")
        store_format = self.util.config_val("FEATS", "store_format", "pkl")
        storage = f"{store}{self.name}.{store_format}"
        extract = self.util.config_val("FEATS", "needs_feature_extraction", False)
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        text_column = "text"
        if text_column not in self.data_df.columns:
            self.util.error(f"no {text_column} column found in data")
        if extract or no_reuse or not os.path.isfile(storage):
            if not self.model_initialized:
                self.init_model()
            self.util.debug(
                "extracting textclassifier results, this might take a while..."
            )
            emb_series = pd.Series(index=self.data_df.index, dtype=object)
            for idx, row in tqdm(self.data_df.iterrows(), total=len(self.data_df)):
                file = idx[0]
                text = row[text_column]
                emb = self.get_results(text, file)
                emb_series[idx] = emb
            self.df = pd.DataFrame(emb_series.values.tolist(), index=self.data_df.index)
            # add the column names:  classification_winner and then all candidate labels
            col_names = ["classification_winner"] + self.candidate_labels
            self.df.columns = col_names
            self.util.write_store(self.df, storage, store_format)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "false"
            except KeyError:
                pass
        else:
            self.util.debug(f"reusing extracted {storage} results")
            self.df = self.util.get_store(storage, store_format)
        if self.df.isnull().values.any():
            self.util.error(f"got nan: {self.df.shape} {self.df.isnull().sum().sum()}")
        return self.df

    def get_results(self, text, file):
        r"""Extract classifier results from text.

        A classifier result looks like this:
        sequence_to_classify = "Hurra ich habe im Lotto gewonnen!"
        candidate_labels = ["valence", "arousal", "dominance"]
        classifier(sequence_to_classify, candidate_labels)
        -> {'sequence': '...', 'labels': ['valence', 'arousal', 'dominance'],
            'scores': [0.8, 0.15, 0.05]}

        We want to return a list with the winner label as the first element
        and the scores (probabilities) for the other labels following.
        The order of the scores is the same as in candidate_labels.
        """
        result = self.model(text, self.candidate_labels)
        # Create a mapping from label to score
        label_to_score = dict(zip(result["labels"], result["scores"]))
        # Get the winner label (first element in result['labels'] is highest scoring)
        winner_label = result["labels"][0]
        # Get scores in the same order as candidate_labels
        scores = [label_to_score[label] for label in self.candidate_labels]
        # Return winner label as first element, followed by scores
        result = [winner_label] + scores
        return result

    def extract_sample(self, text):
        self.init_model()
        feats = self.get_results(text, "no file")
        return feats
