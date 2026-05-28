# feats_audmodel.py
import os
import uuid

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import audeer
import audformat
import audinterface
import audiofile
import audmodel
import audonnx

from nkululeko.feat_extract.featureset import Featureset
import nkululeko.glob_conf as glob_conf


class AudmodelSet(Featureset):
    """Generic audmodel import.

    https://audeering.github.io/audmodel/index.html
    """

    def __init__(self, name, data_df, feats_type="audmodel"):
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
        self.model_interface = audinterface.Feature(
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
            df = pd.DataFrame()
            for file, start, end in tqdm(self.data_df.index):
                signal, sr = None, None
                try:
                    if end == pd.NaT:
                        signal, sr = audiofile.read(file, offset=start)
                    else:
                        signal, sr = audiofile.read(
                            file, duration=end - start, offset=start
                        )
                    f_df = self.extract_sample_df(signal, sr, (file, start, end))
                except Exception as ror:
                    tmp_file_name = f"tmp_{uuid.uuid4()}.wav"
                    if signal is not None and sr is not None:
                        try:
                            audiofile.write(tmp_file_name, signal, sr)
                            self.util.debug(f"segment from {file} written to {tmp_file_name} for debugging.")
                        except Exception as write_err:
                            self.util.warn(f"could not write debug audio: {write_err}")
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
        model_id = self.util.config_val("FEATS", "audmodel.id", "audmodel")
        segment_cache = audeer.mkdir(
            audeer.path(self.util.get_path("cache"), model_id)
        )
        start = index_tuple[1].total_seconds() if index_tuple[1] is not pd.NaT else 0
        end = index_tuple[2].total_seconds() if index_tuple[2] is not pd.NaT else -1
        cache_name = f"{audeer.basename_wo_ext(index_tuple[0])}_{start}_{end}"
        cache_path = audeer.path(segment_cache, cache_name + ".csv")
        if os.path.isfile(cache_path):
            df_part = audformat.utils.read_csv(cache_path)
        else:
            features = self.extract_sample(signal, sr)
            index_part = audformat.segmented_index(
                index_tuple[0], index_tuple[1], index_tuple[2]
            )
            df_part = pd.DataFrame([features], index=index_part)
            df_part.to_csv(cache_path)

        return df_part

    def extract_sample(self, signal, sr):
        if not self.model_loaded:
            self._load_model()
        result = self.model_interface.process_signal(signal, sr)
        return np.asarray(result.values).flatten()
