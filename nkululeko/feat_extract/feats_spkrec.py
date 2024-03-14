# feats_hubert.py
# Speaker embedding feature extractor for Nkululeko
# https://huggingface.co/speechbrain/spkrec-xvect-voxceleb
# supported feat_type:
# "spkrec-xvect-voxceleb", "spkrec-ecapa-voxceleb", "spkrec-resnet-voxceleb"


import os

import nkululeko.glob_conf as glob_conf
import pandas as pd
import torch
import torchaudio
from nkululeko.feat_extract.featureset import Featureset
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm

# from transformers import HubertModel, Wav2Vec2FeatureExtractor


class Spkrec(Featureset):
    """Class to extract SpeechBrain Speaker Embedding embedding)"""

    def __init__(self, name, data_df, feat_type):
        """Constructor. is_train is needed to distinguish from test/dev sets,
        because they use the codebook from the training"""
        super().__init__(name, data_df)
        # check if device is not set, use cuda if available
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        self.classifier_initialized = False
        if feat_type == None:
            self.feat_type = "spkrec-ecapa-voxceleb"
        self.feat_type = feat_type

    def init_model(self):
        # load model
        self.util.debug("loading Spkrec model...")

        model_path = self.util.config_val(
            "FEATS", "Spkrec.model", f"speechbrain/{self.feat_type}"
        )
        self.classifier = EncoderClassifier.from_hparams(model_path)
        print(f"intialized SB model on {self.device}")
        # self.classifier.eval()
        self.classifier_initialized = True

    def extract(self):
        """Extract the features or load them from disk if present."""
        store = self.util.get_path("store")
        storage = f"{store}{self.name}.pkl"
        extract = self.util.config_val(
            "FEATS", "needs_feature_extraction", False
        )
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        if extract or no_reuse or not os.path.isfile(storage):
            if not self.classifier_initialized:
                self.init_model()
            self.util.debug(
                "extracting Spkrec embeddings, this might take a while..."
            )
            emb_series = pd.Series(index=self.data_df.index, dtype=object)
            length = len(self.data_df.index)
            for idx, (file, start, end) in enumerate(
                tqdm(self.data_df.index.to_list())
            ):
                signal, sampling_rate = torchaudio.load(
                    file,
                    frame_offset=int(start.total_seconds() * 16000),
                    num_frames=int((end - start).total_seconds() * 16000),
                )
                assert sampling_rate == 16000
                # check if signal is stereo, if so, take first channel
                if signal.shape[0] == 2:
                    signal = signal[0]
                emb = self.get_embeddings(signal, sampling_rate, file)
                # fill series with embeddings
                emb_series.iloc[idx] = emb
            # print(f"emb_series shape: {emb_series.shape}")
            self.df = pd.DataFrame(
                emb_series.values.tolist(), index=self.data_df.index
            )
            print(f"df shape: {self.df.shape}")
            self.df.to_pickle(storage)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "false"
            except KeyError:
                pass
        else:
            self.util.debug(f"reusing extracted {self.feat_type} embeddings")
            self.df = pd.read_pickle(storage)
            if self.df.isnull().values.any():
                nanrows = self.df.columns[self.df.isna().any()].tolist()
                print(nanrows)
                self.util.error(
                    f"got nan: {self.df.shape} {self.df.isnull().sum().sum()}"
                )

    def get_embeddings(self, signal, sampling_rate, file):
        r"""Extract embeddings from raw audio signal."""
        # classifier = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")
        try:
            # with torch.no_grad():
            y = self.classifier.encode_batch(signal)
            y = y.squeeze().detach().cpu().numpy()
        except RuntimeError as re:
            print(str(re))
            self.util.error(f"couldn't extract file: {file}")
        # print(f"y ravel shape: {y.ravel().shape}")
        # if y.ravel().shape[0] != 192:
        #     self.util.error(f"got wrong embedding size: {y.ravel().shape} from file: {file}")
        # print(f"y: {y}")
        return y.ravel()

    def extract_sample(self, signal, sr):
        self.init_model()
        feats = self.get_embeddings(signal, sr, "no file")
        return feats
