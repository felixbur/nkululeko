# feats_audmodel.py
import os

import numpy as np
import torch

import audeer
import audinterface
import audmodel
import audonnx

from nkululeko.feat_extract.featureset import Featureset
import nkululeko.glob_conf as glob_conf


class AudmodelSet(Featureset):
    """Generic audmodel import.

    https://audeering.github.io/audmodel/index.html
    """

    def __init__(self, name, data_df, feats_type):
        super().__init__(name, data_df, feats_type)
        self.model_loaded = False
        self.feats_type = feats_type

    def _load_model(self):
        model_id = self.util.config_val("FEATS", "audmodel.id", False)
        if model_id is False:
            self.util.error(
                "Please set the audmodel.id in the config file to the model you want to use."
            )
        self.embeddings_name = self.util.config_val(
            "FEATS", "audmodel.embeddings_name", "hidden_states"
        )
        model_root = audmodel.load(model_id)
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
                self.model.labels(self.embeddings_name),
                process_func=self.model,
                process_func_args={
                    "outputs": self.embeddings_name,
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
        if self.model is None:
            self.__init__("na", None)
        result = self.model(signal, sr)
        return np.asarray(result[self.embeddings_name].flatten())
