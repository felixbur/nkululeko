# feats_wav2vec2.py
# feat_types example = wav2vec2-large-robust-ft-swbd-300h
import os
from tqdm import tqdm
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import transformers
from nkululeko.feat_extract.featureset import Featureset
import nkululeko.glob_conf as glob_conf


class Wav2vec2(Featureset):
    """Class to extract wav2vec2 embeddings"""

    def __init__(self, name, data_df, feat_type):
        """Constructor. is_train is needed to distinguish from test/dev sets, because they use the codebook from the training"""
        super().__init__(name, data_df)
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        self.model_initialized = False
        if feat_type == "wav2vec" or feat_type == "wav2vec2":
            self.feat_type = "wav2vec2-large-robust-ft-swbd-300h"
        else:
            self.feat_type = feat_type

    def init_model(self):
        # load model
        self.util.debug("loading wav2vec2 model...")
        model_path = self.util.config_val(
            "FEATS", "wav2vec2.model", f"facebook/{self.feat_type}"
        )
        config = transformers.AutoConfig.from_pretrained(model_path)
        layer_num = config.num_hidden_layers
        hidden_layer = int(self.util.config_val("FEATS", "wav2vec2.layer", "0"))
        config.num_hidden_layers = layer_num - hidden_layer
        self.util.debug(f"using hidden layer #{config.num_hidden_layers}")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = Wav2Vec2Model.from_pretrained(model_path, config=config).to(
            self.device
        )
        print(f"intialized Wav2vec model on {self.device}")
        self.model.eval()
        self.model_initialized = True

    def extract(self):
        """Extract the features or load them from disk if present."""
        store = self.util.get_path("store")
        storage = f"{store}{self.name}.pkl"
        extract = self.util.config_val("FEATS", "needs_feature_extraction", False)
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        if extract or no_reuse or not os.path.isfile(storage):
            if not self.model_initialized:
                self.init_model()
            self.util.debug(
                "extracting wav2vec2 embeddings, this might take a while..."
            )
            emb_series = pd.Series(index=self.data_df.index, dtype=object)
            for idx, (file, start, end) in enumerate(
                tqdm(self.data_df.index.to_list())
            ):
                signal, sampling_rate = torchaudio.load(
                    file,
                    frame_offset=int(start.total_seconds() * 16000),
                    num_frames=int((end - start).total_seconds() * 16000),
                )
                assert sampling_rate == 16000, f"got {sampling_rate} instead of 16000"
                emb = self.get_embeddings(signal, sampling_rate, file)
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
            self.util.debug("reusing extracted wav2vec2 embeddings")
            self.df = pd.read_pickle(storage)
            if self.df.isnull().values.any():
                nanrows = self.df.columns[self.df.isna().any()].tolist()
                # print(nanrows)
                self.util.error(
                    f"got nan: {self.df.shape} {self.df.isnull().sum().sum()}"
                )

    def get_embeddings(self, signal, sampling_rate, file):
        r"""Extract embeddings from raw audio signal."""
        try:
            with torch.no_grad():
                # run through processor to normalize signal
                # always returns a batch, so we just get the first entry
                # then we put it on the device
                y = self.processor(signal, sampling_rate=sampling_rate)
                y = y["input_values"][0]
                y = torch.from_numpy(y.reshape(1, -1)).to(self.device)
                # print(y.shape)
                # run through model
                # first entry contains hidden state
                y = self.model(y)[0]

                # pool result and convert to numpy
                y = torch.mean(y, dim=1)
                y = y.detach().cpu().numpy()
        except RuntimeError as re:
            print(str(re))
            self.util.error(f"couldn't extract file: {file}")
        # print(f"y flattened shape: {y.ravel().shape}")
        return y.ravel()

    def extract_sample(self, signal, sr):
        self.init_model()
        feats = self.get_embeddings(signal, sr, "no file")
        return feats
