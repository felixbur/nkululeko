# feats_audmodel_dim.py
from nkululeko.feat_extract.featureset import Featureset
import os
import pandas as pd
import audeer
import nkululeko.glob_conf as glob_conf
import audonnx
import numpy as np
import audinterface


class AgenderAgenderSet(Featureset):
    """
    Age and gender predictions from the wav2vec2. based model finetuned on agender, described in the paper
    "Speech-based Age and Gender Prediction with Transformers"
    https://arxiv.org/abs/2306.16962
    """

    def __init__(self, name, data_df):
        super().__init__(name, data_df)
        self.model_loaded = False

    def _load_model(self):
        model_url = "https://zenodo.org/record/7761387/files/w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip"
        model_root = self.util.config_val(
            "FEATS", "agender.model", "./audmodel_agender/"
        )
        if not os.path.isdir(model_root):
            cache_root = audeer.mkdir("cache")
            model_root = audeer.mkdir(model_root)
            archive_path = audeer.download_url(model_url, cache_root, verbose=True)
            audeer.extract_archive(archive_path, model_root)
        device = self.util.config_val("MODEL", "device", "cpu")
        self.model = audonnx.load(model_root, device=device)
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        self.util.debug(
            f"initialized agender model with {pytorch_total_params} parameters in total"
        )
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
        sampling_rate = 16000
        if no_reuse or extract or not os.path.isfile(storage):
            self.util.debug(
                "extracting agender model age and gender, this might take a" " while..."
            )
            if not self.model_loaded:
                self._load_model()
            outputs = ["logits_age", "logits_gender"]
            logits = audinterface.Feature(
                self.model.labels(outputs),
                process_func=self.model,
                process_func_args={
                    "outputs": outputs,
                    "concat": True,
                },
                sampling_rate=sampling_rate,
                resample=True,
                verbose=True,
            )
            self.df = logits.process_index(self.data_df.index)
            self.util.write_store(self.df, storage, store_format)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "False"
            except KeyError:
                pass
        else:
            self.util.debug("reusing extracted audmodel features.")
            self.df = self.util.get_store(storage, store_format)

    def extract_sample(self, signal, sr):
        result = self.model(signal, sr)
        return np.asarray(result["hidden_states"].flatten())
