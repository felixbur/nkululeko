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
        self.hidden_layer = int(self.util.config_val("FEATS", "wav2vec2.layer", "0"))
        self.use_torch = False

    def _load_model(self):
        model_id = self.util.config_val("FEATS", "audmodel.id", False)
        if model_id is False:
            self.util.error(
                "Please set the audmodel.id in the config file to the model you want to use."
            )
        self.embeddings_name = self.util.config_val(
            "FEATS", "audmodel.embeddings_name", "hidden_states"
        )
        self.util.debug(f"loading audmodel {model_id}, this might take a while...")
        model_root = audmodel.load(model_id)
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)

        if self.hidden_layer != 0 or not os.path.isfile(
            os.path.join(model_root, "model.onnx")
        ):
            self._load_torch_model(model_root)
        else:
            self._load_onnx_model(model_root)

        self.model_loaded = True

    def _load_onnx_model(self, model_root):
        self.util.debug("loading as ONNX model")
        self.model = audonnx.load(model_root, device=self.device)
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
        self.use_torch = False

    def _load_torch_model(self, model_root):
        import transformers
        self.util.debug("loading as torch model")
        config = transformers.AutoConfig.from_pretrained(model_root)
        if self.hidden_layer != 0:
            num_layers = getattr(config, "num_hidden_layers", None)
            if num_layers is not None:
                adjusted = num_layers - self.hidden_layer
                self.util.debug(
                    f"using layer #{adjusted} of {num_layers} "
                    f"(audmodel.layer={self.hidden_layer} from end)"
                )
                config.num_hidden_layers = adjusted
            else:
                self.util.warn(
                    "model config has no num_hidden_layers; audmodel.layer will be ignored"
                )
        try:
            self.processor = transformers.AutoFeatureExtractor.from_pretrained(f"{model_root}/config.json")
        except Exception:
            base = getattr(config, "_name_or_path", None)
            if base:
                self.util.debug(
                    f"no preprocessor config in model dir; loading feature extractor from {base}"
                )
                self.processor = transformers.AutoFeatureExtractor.from_pretrained(base)
            else:
                self.util.debug("no preprocessor config found; using default Wav2Vec2FeatureExtractor")
                self.processor = transformers.Wav2Vec2FeatureExtractor(
                    feature_size=1,
                    sampling_rate=16000,
                    padding_value=0.0,
                    do_normalize=True,
                    return_attention_mask=False,
                )
        self.model = transformers.AutoModel.from_pretrained(
            model_root, config=config
        ).to(self.device)
        self.model.eval()
        self.use_torch = True

    def extract(self):
        """Extract the features based on the initialized dataset or re-open them when found on disk."""
        store = self.util.get_path("store")
        store_format = self.util.config_val("FEATS", "store_format", "pkl")
        if self.hidden_layer == 0:
            storage = f"{store}{self.name}.{store_format}"
        else:
            storage = f"{store}{self.name}_l{self.hidden_layer}.{store_format}"
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
        layer_suffix = f"_l{self.hidden_layer}" if self.hidden_layer != 0 else ""
        cache_name = f"{audeer.basename_wo_ext(index_tuple[0])}_{start}_{end}{layer_suffix}"
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
        if not self.use_torch:
            result = self.model_interface.process_signal(signal, sr)
            return np.asarray(result.values).flatten()
        # Torch path: resample if needed, pool last hidden state
        import torchaudio
        signal_tensor = torch.from_numpy(signal).float()
        if sr != 16000:
            signal_tensor = torchaudio.functional.resample(signal_tensor, sr, 16000)
        if signal_tensor.ndim > 1:
            signal_tensor = signal_tensor.mean(dim=0)
        with torch.no_grad():
            inputs = self.processor(
                signal_tensor.numpy(), sampling_rate=16000, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            y = outputs.last_hidden_state  # [batch, time, hidden_size]
            y = torch.mean(y, dim=1)       # pool over time -> [batch, hidden_size]
            return y.detach().cpu().numpy().flatten()
