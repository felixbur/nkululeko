# feats_clap.py

import os

import audiofile
import laion_clap
import pandas as pd
from tqdm import tqdm

import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.featureset import Featureset


class ClapSet(Featureset):
    """Class to extract laion's clap embeddings (https://github.com/LAION-AI/CLAP)"""

    def __init__(self, name, data_df, feats_type):
        """Constructor. is_train is needed to distinguish from test/dev sets, because they use the codebook from the training"""
        super().__init__(name, data_df, feats_type)
        self.device = self.util.config_val("MODEL", "device", "cpu")
        self.model_initialized = False
        self.feat_type = feats_type

    def init_model(self):
        # load model
        self.util.debug("loading clap model...")
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()  # download the default pretrained checkpoint.
        print("loaded clap model")

    def extract(self):
        """Extract the features or load them from disk if present."""
        store = self.util.get_path("store")
        store_format = self.util.config_val("FEATS", "store_format", "pkl")
        storage = f"{store}{self.name}.{store_format}"
        extract = self.util.config_val("FEATS", "needs_feature_extraction", False)
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        if extract or no_reuse or not os.path.isfile(storage):
            if not self.model_initialized:
                self.init_model()
            self.util.debug("extracting clap embeddings, this might take a while...")
            emb_series = pd.Series(index=self.data_df.index, dtype=object)
            length = len(self.data_df.index)
            for idx, (file, start, end) in enumerate(
                tqdm(self.data_df.index.to_list())
            ):
                signal, sampling_rate = audiofile.read(
                    file,
                    offset=start.total_seconds(),
                    duration=(end - start).total_seconds(),
                    always_2d=True,
                )
                emb = self.get_embeddings(signal, sampling_rate)
                emb_series[idx] = emb
            self.df = pd.DataFrame(emb_series.values.tolist(), index=self.data_df.index)
            self.util.write_store(self.df, storage, store_format)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "false"
            except KeyError:
                pass
        else:
            self.util.debug("reusing extracted wav2vec2 embeddings")
            self.df = self.util.get_store(storage, store_format)
            if self.df.isnull().values.any():
                nanrows = self.df.columns[self.df.isna().any()].tolist()
                print(nanrows)
                self.util.error(
                    f"got nan: {self.df.shape} {self.df.isnull().sum().sum()}"
                )

    def get_embeddings(self, signal, sampling_rate):
        tmp_audio_name = ["clap_audio_tmp.wav"]
        audiofile.write(tmp_audio_name[0], signal, 48000)
        audio_embed = self.model.get_audio_embedding_from_filelist(
            x=tmp_audio_name, use_tensor=False
        )
        return audio_embed[0]

    def extract_sample(self, signal, sr):
        self.init_model()
        feats = self.get_embeddings(signal, sr)
        return feats
