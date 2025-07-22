import os

import pandas as pd
from tqdm import tqdm
import transformers
import torch 
from transformers import BertTokenizer, BertModel

from nkululeko.feat_extract.featureset import Featureset
import nkululeko.glob_conf as glob_conf


class Bert(Featureset):
    """Class to extract bert embeddings"""

    def __init__(self, name, data_df, feat_type):
        """Constructor.

        If_train is needed to distinguish from test/dev sets,
        because they use the codebook from the training
        """
        super().__init__(name, data_df, feat_type)
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        self.model_initialized = False
        if feat_type == "bert":
            self.feat_type = "bert-base-uncased"
        else:
            self.feat_type = feat_type

    def init_model(self):
        # load model
        self.util.debug(f"loading {self.feat_type} model...")
        model_path = self.util.config_val(
            "FEATS", "bert.model", f"google-bert/{self.feat_type}"
        )
        config = transformers.AutoConfig.from_pretrained(model_path)
        layer_num = config.num_hidden_layers
        hidden_layer = int(self.util.config_val("FEATS", "bert.layer", "0"))
        config.num_hidden_layers = layer_num - hidden_layer
        self.util.debug(f"using hidden layer #{config.num_hidden_layers}")

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path, config=config).to(
            self.device
        )
        print(f"initialized {self.feat_type} model on {self.device}")
        self.model.eval()
        self.model_initialized = True

    def extract(self):
        """Extract the features or load them from disk if present."""
        store = self.util.get_path("store")
        storage = os.path.join(store, f"{self.name}.pkl")
        extract = self.util.config_val("FEATS", "needs_feature_extraction", False)
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        if extract or no_reuse or not os.path.isfile(storage):
            if not self.model_initialized:
                self.init_model()
            self.util.debug(
                f"extracting {self.feat_type} embeddings, this might take a while..."
            )
            emb_series = pd.Series(index=self.data_df.index, dtype=object)
            for idx, row in tqdm(self.data_df.iterrows(), total=len(self.data_df)):
                file = idx[0]
                text = row['text']
                emb = self.get_embeddings(text, file)
                emb_series[idx] = emb
            # print(f"emb_series shape: {emb_series.shape}")
            self.df = pd.DataFrame(emb_series.values.tolist(), index=self.data_df.index)
            # print(f"df shape: {self.df.shape}")
            self.df.to_pickle(storage)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "false"
            except KeyError:
                pass
        else:
            self.util.debug(f"reusing extracted {self.feat_type} embeddings")
            self.df = pd.read_pickle(storage)
            if self.df.isnull().values.any():
                self.util.error(
                    f"got nan: {self.df.shape} {self.df.isnull().sum().sum()}"
                )

    def get_embeddings(self, text, file):
        r"""Extract embeddings from raw audio signal."""
        try:
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                # mean pooling 
                y = torch.mean(outputs[0], dim=1)
                y = y.ravel()
        except RuntimeError as re:
            print(str(re))
            self.util.error(f"couldn't extract file: {file}")
            y = None
        if y is None:
            return None
        return y.detach().cpu().numpy()

    def extract_sample(self, text):
        self.init_model()
        feats = self.get_embeddings(text, "no file")
        return feats
