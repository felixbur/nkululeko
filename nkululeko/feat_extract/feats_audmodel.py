# feats_audmodel.py
import ast
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


def parse_embeddings_names(raw):
    """Parse FEATS.audmodel.embeddings_name into a list of output names.

    Accepts either a single bare name (e.g. "valence", the back-compat
    default) or a list literal (e.g. "['arousal', 'valence']") for
    multi-head extraction (GH #459): a bare word isn't valid Python literal
    syntax, so ast.literal_eval raises and we fall back to treating `raw`
    as the one name as-is.
    """
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        parsed = raw
    if isinstance(parsed, (list, tuple)):
        if len(parsed) == 0:
            raise ValueError("FEATS.audmodel.embeddings_name is an empty list")
        return [str(p) for p in parsed]
    return [str(parsed)]


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
        raw_embeddings_name = self.util.config_val(
            "FEATS", "audmodel.embeddings_name", "hidden_states"
        )
        try:
            self.embeddings_names = parse_embeddings_names(raw_embeddings_name)
        except ValueError as error:
            self.util.error(str(error))
        # kept for any external/older code that still reads the singular
        # attribute; only meaningful for the (default) single-head case.
        self.embeddings_name = self.embeddings_names[0]
        self.util.debug(f"loading audmodel {model_id}, this might take a while...")
        model_root = audmodel.load(model_id)
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)

        if len(self.embeddings_names) > 1 and (
            self.hidden_layer != 0
            or not os.path.isfile(os.path.join(model_root, "model.onnx"))
        ):
            self.util.error(
                "FEATS.audmodel.embeddings_name as a list (multi-head "
                "extraction) is only supported for ONNX models with named "
                "output heads; this model would use the torch path "
                "(audmodel.wav2vec2.layer != 0, or no model.onnx present), "
                "which has no concept of named output heads."
            )

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
        providers = self.model.sess.get_providers()
        if self.device == "cuda" and "CUDAExecutionProvider" not in providers:
            self.util.warn(
                "FEATS.audmodel: requested CUDA but onnxruntime fell back to "
                f"{providers} -- extraction will be much slower. Check that "
                "cuDNN 9 is on LD_LIBRARY_PATH (torch ships it under "
                "<venv>/lib/.../site-packages/nvidia/cudnn/lib)."
            )
        heads = self.embeddings_names
        multi_head = len(heads) > 1
        self._head_label_counts = [len(self.model.labels(h)) for h in heads]
        feature_names = self.model.labels(heads if multi_head else heads[0])
        process_func_args = {"outputs": heads if multi_head else heads[0]}
        if multi_head:
            process_func_args["concat"] = True
        self.model_interface = audinterface.Feature(
            feature_names,
            process_func=self.model,
            process_func_args=process_func_args,
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
        # audmodel.id and audmodel.embeddings_name are both part of the
        # cache key (GH #459 and its review): self.name alone (e.g.
        # "mydb_train_audmodel", built in feature_extractor.py from the
        # database name + feats_type, never the model) is identical
        # regardless of which audmodel.id/embeddings_name are configured,
        # so without both here, changing either one silently reuses the
        # previous model/head's cached values.
        model_id = self.util.config_val("FEATS", "audmodel.id", "audmodel")
        raw_embeddings_name = self.util.config_val(
            "FEATS", "audmodel.embeddings_name", "hidden_states"
        )
        embeddings_name = "-".join(parse_embeddings_names(raw_embeddings_name))
        if self.hidden_layer == 0:
            storage = f"{store}{self.name}_{model_id}_{embeddings_name}.{store_format}"
        else:
            storage = (
                f"{store}{self.name}_{model_id}_{embeddings_name}_l{self.hidden_layer}."
                f"{store_format}"
            )
        if self._needs_extraction(storage):
            self.util.debug(
                "extracting audmodel embeddings, this might take a while..."
            )
            df = pd.DataFrame()
            for file, start, end in tqdm(self.data_df.index):
                signal, sr = None, None
                try:
                    if pd.isna(end):
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
                    # Keep this row as missing (NaN once concatenated with
                    # the columns other rows have) instead of aborting the
                    # whole extraction (review on #468): a single bad file
                    # would otherwise abort every downstream consumer,
                    # including ap_audmodel.py's PREDICT target, whose
                    # all-zero-row "unknown" fallback could never actually
                    # be reached as long as this raised here.
                    self.util.warn(f"audmodel extraction failed on file {file}: {ror}")
                    f_df = pd.DataFrame(
                        index=audformat.segmented_index(file, start, end)
                    )
                df = pd.concat([df, f_df])
            self.df = df
            self.util.write_store(self.df, storage, store_format)
            try:
                self.context.config["DATA"]["needs_feature_extraction"] = "False"
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
        # audmodel.embeddings_name is part of the cache key too (GH #459),
        # same reasoning as the whole-dataset storage path in extract().
        raw_embeddings_name = self.util.config_val(
            "FEATS", "audmodel.embeddings_name", "hidden_states"
        )
        embeddings_name = "-".join(parse_embeddings_names(raw_embeddings_name))
        segment_cache = audeer.mkdir(
            audeer.path(self.util.get_path("cache"), model_id, embeddings_name)
        )
        layer_suffix = f"_l{self.hidden_layer}" if self.hidden_layer != 0 else ""
        cache_path = self._sample_cache_path(
            segment_cache, index_tuple, suffix=layer_suffix
        )
        if os.path.isfile(cache_path):
            df_part = self._read_sample_cache(cache_path)
        else:
            features, labels = self._extract_sample_with_labels(signal, sr)
            index_part = audformat.segmented_index(
                index_tuple[0], index_tuple[1], index_tuple[2]
            )
            df_part = pd.DataFrame([features], columns=labels, index=index_part)
            df_part.to_csv(cache_path)

        return df_part

    def extract_sample(self, signal, sr):
        """Extract one sample's features as a plain array.

        Kept to the generic Featureset contract (a plain feature array, no
        labels) since this is called directly by other code paths, e.g.
        nkululeko.predict's generic feature-extractor dispatch, which runs
        arithmetic (mean/std) on the result -- see
        _extract_sample_with_labels() for the labeled variant used by
        extract_sample_df().
        """
        features, _ = self._extract_sample_with_labels(signal, sr)
        return features

    def _extract_sample_with_labels(self, signal, sr):
        """Extract one sample's features, with column labels.

        Returns:
            tuple: (features, labels) -- labels are the real output names
            (e.g. "sadness", "arousal") for the ONNX path, since
            audinterface.Feature already carries them as result.columns
            from the audmodel.labels() passed in at construction (GH #459);
            generic `{embeddings_name}_{i}` names for the torch path, which
            has no per-dimension names to offer.
        """
        if not self.model_loaded:
            self._load_model()
        if not self.use_torch:
            result = self.model_interface.process_signal(signal, sr)
            labels = self._prefixed_head_labels(result.columns)
            return np.asarray(result.values).flatten(), labels
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
            features = y.detach().cpu().numpy().flatten()
            labels = [f"{self.embeddings_name}_{i}" for i in range(len(features))]
            return features, labels

    def _prefixed_head_labels(self, columns):
        """Prefix each ONNX output column with its owning head's name.

        With multi-head extraction (GH #459 item 3), `columns` is every
        configured head's labels concatenated in head order with no
        indication of where one head's columns end and the next begin --
        `self._head_label_counts` (each head's own label count, recorded at
        load time in the same order) recovers the boundaries. A
        one-dimensional head's single column is the head name itself
        (not prefixed with itself), matching the prior single-head
        behavior exactly when there's only one configured head.
        """
        labels = []
        pos = 0
        for head, count in zip(self.embeddings_names, self._head_label_counts):
            head_columns = columns[pos : pos + count]
            pos += count
            if count == 1:
                labels.append(head)
            else:
                labels.extend(f"{head}_{c}" for c in head_columns)
        return labels
