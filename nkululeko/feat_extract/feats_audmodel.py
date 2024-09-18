# feats_audmodel.py
import os

import audeer
import audinterface
import audonnx
import numpy as np
import torch

import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.featureset import Featureset


class AudmodelSet(Featureset):
    """Embeddings from the wav2vec2 based model finetuned on MSPPodcast emotions.

    Described in the paper:
    "Dawn of the transformer era in speech emotion recognition: closing the valence gap"
    https://arxiv.org/abs/2203.07378.
    """

    def __init__(self, name, data_df, feats_type):
        super().__init__(name, data_df, feats_type)
        self.model_loaded = False
        self.feats_type = feats_type

    def _load_model(self):
        model_url = "https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip"
        model_root = self.util.config_val("FEATS", "aud.model", "./audmodel/")
        if not os.path.isdir(model_root):
            cache_root = audeer.mkdir("cache")
            model_root = audeer.mkdir(model_root)
            archive_path = audeer.download_url(model_url, cache_root, verbose=True)
            audeer.extract_archive(archive_path, model_root)
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        device = self.util.config_val("MODEL", "device", cuda)
        self.model = audonnx.load(model_root, device=device)
        self.model_loaded = True

    def extract(self):
        """Extract the features based on the initialized dataset or re-open them when found on disk."""
        store = self.util.get_path("store")
        store_format = self.util.config_val("FEATS", "store_format", "pkl")
        storage = f"{store}{self.name}.{store_format}"
        extract = eval(
            self.util.config_val("FEATS", "needs_feature_extraction", "False")
        )
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        if no_reuse or extract or not os.path.isfile(storage):
            self.util.debug(
                "extracting audmodel embeddings, this might take a while..."
            )
            if not self.model_loaded:
                self._load_model()
            hidden_states = audinterface.Feature(
                self.model.labels("hidden_states"),
                process_func=self.model,
                process_func_args={
                    "outputs": "hidden_states",
                },
                sampling_rate=16000,
                resample=True,
                num_workers=self.n_jobs,
                verbose=True,
            )
            self.df = hidden_states.process_index(self.data_df.index)
            self.util.write_store(self.df, storage, store_format)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "False"
            except KeyError:
                pass
        else:
            self.util.debug("reusing extracted audmodel features.")
            self.df = self.util.get_store(storage, store_format)

    def extract_sample(self, signal, sr):
        if self.model == None:
            self.__init__("na", None)
        result = self.model(signal, sr)
        return np.asarray(result["hidden_states"].flatten())
