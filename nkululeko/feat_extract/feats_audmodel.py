# feats_audmodel.py
import ast
import hashlib
import os

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

from nkululeko.feat_extract.featureset import EXTRACTION_ERRORS, Featureset


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
        heads = [str(p) for p in parsed]
        seen = set()
        duplicates = [h for h in heads if (h in seen) or seen.add(h)]
        if duplicates:
            raise ValueError(
                "FEATS.audmodel.embeddings_name has duplicate head(s): "
                f"{sorted(set(duplicates))}"
            )
        return heads
    return [str(parsed)]


def embeddings_cache_key(raw):
    """Collision-free cache-key component for FEATS.audmodel.embeddings_name.

    Single head: the bare head name, kept human-readable (unchanged from
    before multi-head support existed). Multiple heads: joining the names
    with a plain separator like "-" is ambiguous -- ['foo-bar', 'baz'] and
    ['foo', 'bar-baz'] would both join to "foo-bar-baz" (review on #468),
    silently reusing a different head selection's cached features. Hashing
    the ordered list instead (joined on an ASCII unit separator a real head
    name won't contain, so the pre-hash string itself is already
    unambiguous) avoids that, and sidesteps escaping concerns for using
    arbitrary head names in a filesystem path.
    """
    heads = parse_embeddings_names(raw)
    if len(heads) == 1:
        return heads[0]
    digest = hashlib.sha256("\x1f".join(heads).encode()).hexdigest()[:16]
    return f"multihead-{digest}"


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
                "(FEATS.wav2vec2.layer != 0, or no model.onnx present), "
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
        requested_cuda = self.device == "cuda" or self.device.startswith("cuda:")
        if requested_cuda and "CUDAExecutionProvider" not in providers:
            self.util.warn(
                "FEATS.audmodel: requested CUDA but onnxruntime fell back to "
                f"{providers} -- extraction will be much slower. Check that "
                "cuDNN 9 is on LD_LIBRARY_PATH (torch ships it under "
                "<venv>/lib/.../site-packages/nvidia/cudnn/lib)."
            )
        heads = self.embeddings_names
        invalid = [h for h in heads if h not in self.model.outputs]
        if invalid:
            valid = sorted(self.model.outputs.keys())
            self.util.error(
                f"FEATS.audmodel.embeddings_name: unknown output head(s) "
                f"{invalid}; this model's valid output names are: {valid}"
            )
        multi_head = len(heads) > 1
        self._head_label_counts = [len(self.model.labels(h)) for h in heads]
        feature_names = self.model.labels(heads if multi_head else heads[0])
        prefixed_names = self._prefixed_head_labels(feature_names)
        seen = set()
        collisions = sorted({n for n in prefixed_names if n in seen or seen.add(n)})
        if collisions:
            # Two different heads' generated column names (head name for a
            # one-dimension head, "{head}_{label}" otherwise) can still
            # collide even with unique, non-duplicate head names -- e.g. a
            # one-dimension head "foo_bar" and a categorical head "foo"
            # with a label "bar" both produce the column "foo_bar". That
            # leaves pred_df with duplicate-named columns, which breaks
            # any single-column selection downstream (AudmodelPredictor's
            # pred_df[head] would return a DataFrame, not a Series).
            self.util.error(
                "FEATS.audmodel.embeddings_name: generated feature column "
                f"name(s) collide across different heads: {collisions}. "
                "Choose head names that don't produce the same prefixed "
                "column (e.g. a one-dimension head's own name must not "
                "equal another head's '<head>_<label>' combination)."
            )
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
        try:
            embeddings_name = embeddings_cache_key(raw_embeddings_name)
        except ValueError as error:
            self.util.error(str(error))
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
            # Load the model before the per-file loop, not lazily inside
            # it (review on #468): the per-sample cache check in
            # extract_sample_df() is unconditional (unlike the
            # whole-dataset store, it isn't gated by FEATS.no_reuse), so a
            # run where every file already has a cached segment would
            # otherwise never call _load_model() at all, leaving
            # self.embeddings_names/_head_label_counts unset -- which
            # ap_audmodel.py's AudmodelPredictor relies on always being
            # present. Loading eagerly also keeps a genuine config error
            # from _load_model() (e.g. an unknown/duplicate head) from
            # being swallowed by the per-file except clause below:
            # NkululukoError subclasses RuntimeError, one of the
            # EXTRACTION_ERRORS types tolerated there as a per-file miss.
            if not self.model_loaded:
                self._load_model()
            df = pd.DataFrame()
            total = len(self.data_df.index)
            failed = 0
            for file, start, end in tqdm(self.data_df.index):
                try:
                    if pd.isna(end):
                        signal, sr = audiofile.read(file, offset=start)
                    else:
                        signal, sr = audiofile.read(
                            file, duration=end - start, offset=start
                        )
                    f_df = self.extract_sample_df(signal, sr, (file, start, end))
                except EXTRACTION_ERRORS as ror:
                    # Keep this row as missing (NaN once concatenated with
                    # the columns other rows have) instead of aborting the
                    # whole extraction (review on #468): a single bad file
                    # would otherwise abort every downstream consumer,
                    # including ap_audmodel.py's PREDICT target, whose
                    # all-zero-row "unknown" fallback could never actually
                    # be reached as long as this raised here. Only the
                    # shared EXTRACTION_ERRORS types (bad/corrupt audio,
                    # I/O issues) are tolerated this way -- anything else
                    # (e.g. a KeyError/TypeError from a programming bug)
                    # still propagates and aborts, same as before. No debug
                    # audio dump here (review on #468): with a tolerated
                    # failure rate instead of an immediate abort, a large
                    # dataset could otherwise leak many undeleted temp
                    # files; the warning's file/start/end already points
                    # at the original audio for manual inspection.
                    self.util.warn(f"audmodel extraction failed on file {file}: {ror}")
                    f_df = pd.DataFrame(
                        index=audformat.segmented_index(file, start, end)
                    )
                    failed += 1
                df = pd.concat([df, f_df])
            if failed > 0:
                self.util.warn(
                    f"audmodel extraction: {failed}/{total} files failed"
                    f" ({100 * failed / total:.1f}%)"
                )
                fail_threshold = self._get_fail_threshold()
                if total > 0 and failed / total > fail_threshold:
                    # Enforce the same FEATS.fail_threshold contract as
                    # _extract_embeddings_with_error_handling() (review on
                    # #468): keeping individual failures as NaN rows must
                    # not let a store that's entirely (or mostly) empty get
                    # persisted and silently reused as if extraction had
                    # succeeded.
                    self.util.error(
                        f"audmodel extraction failure rate {failed / total:.0%}"
                        f" exceeds threshold {fail_threshold:.0%}"
                    )
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
        try:
            embeddings_name = embeddings_cache_key(raw_embeddings_name)
        except ValueError as error:
            self.util.error(str(error))
        segment_cache = audeer.mkdir(
            audeer.path(self.util.get_path("cache"), model_id, embeddings_name)
        )
        layer_suffix = f"_l{self.hidden_layer}" if self.hidden_layer != 0 else ""
        cache_path = self._sample_cache_path(
            segment_cache, index_tuple, suffix=layer_suffix
        )
        # The per-sample cache key is (file path, start, end) only -- it
        # has no way to detect that the audio at that path changed, so it
        # must still honor FEATS.no_reuse explicitly (review on #468):
        # without this, forcing no_reuse at the whole-dataset level (e.g.
        # ap_audmodel.py's AudmodelPredictor, for every ad-hoc standalone
        # prediction) would still silently return a stale cached sample
        # if a file at the same path/segment was predicted before.
        if os.path.isfile(cache_path) and not self.util.config_val_bool(
            "FEATS", "no_reuse", False
        ):
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
