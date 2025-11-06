# feats_audwav2vec2.py
import os
import pandas as pd
import audeer
import audinterface
import audiofile
import audonnx
import numpy as np
import torch
from tqdm import tqdm
import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.featureset import Featureset


class Audwav2vec2Set(Featureset):
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
                "extracting audwav2vec2 embeddings, this might take a while..."
            )
            df = pd.DataFrame()
            for file, start, end in tqdm(self.data_df.index):
                try:
                    if end == pd.NaT:
                        signal, sr = audiofile.read(file, offset=start)
                    else:
                        signal, sr = audiofile.read(
                            file, duration=end - start, offset=start
                        )
                    f_df = self.extract_sample_df(signal, sr, (file, start, end))
                except Exception as ror:
                    self.util.error(f"error {ror} on file {file}")
                df = pd.concat([df, f_df])
            self.df = df
            self.util.write_store(self.df, storage, store_format)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "False"
            except KeyError:
                pass
        else:
            self.util.debug("reusing extracted audmodel features.")
            self.df = self.util.get_store(storage, store_format)

    def extract_sample_df(self, signal, sr, index_tuple):
        """Extract features for a single sample and return as DataFrame.

        Args:
            signal: Audio signal
            sr: Sample rate
            index_tuple: Tuple of (file, start, end) for DataFrame index

        Returns:
            pd.DataFrame: Features as a single-row DataFrame with proper index
        """
        features = self.extract_sample(signal, sr)
        # Create DataFrame with proper index matching data_df structure
        return pd.DataFrame([features], index=[index_tuple])

    def extract_sample(self, signal, sr):
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
        result = hidden_states.process_signal(signal, sr)
        return np.asarray(result.values).flatten()
