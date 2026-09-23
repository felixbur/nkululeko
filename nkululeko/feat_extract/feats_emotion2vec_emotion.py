# feats_emotion2vec_emotion.py
# emotion2vec's own emotion CLASSIFICATION output (not just embeddings).
#
# feats_emotion2vec.py extracts emotion2vec's raw embedding vector, meant to
# feed a user's own classifier (FEATS.type = ['emotion2vec']). This module
# instead asks FunASR for the emotion2vec "_plus_*" checkpoints' own
# built-in classification output (extract_embedding=False), and returns one
# score column per emotion class -- the same "feature extractor whose
# output columns already are the prediction" convention
# feats_agender_agender.py uses for age/gender. It backs
# nkululeko.autopredict's `--model emotion` target (ap_emotion.py), which
# previously discarded the extracted embedding and always returned
# "neutral" for every sample.
#
# Only the "_plus_*" emotion2vec checkpoints are finetuned for this
# classification task; the plain "emotion2vec_base" universal embedding
# model is not (see FEATS.emotion2vec.model in ini_file.md).

# requirements:
# pip install funasr

import os
import tempfile

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from nkululeko.feat_extract.featureset import Featureset

_MODEL_MAPPING = {
    "emotion2vec_emotion": "emotion2vec/emotion2vec_plus_large",
    "emotion2vec_emotion-base": "emotion2vec/emotion2vec_plus_base",
    "emotion2vec_emotion-seed": "emotion2vec/emotion2vec_plus_seed",
    "emotion2vec_emotion-large": "emotion2vec/emotion2vec_plus_large",
}


def _clean_label(label):
    """Normalize an emotion2vec class label to its plain (English) name.

    FunASR's documented emotion2vec "_plus_*" output puts one label string
    per class in res[0]["labels"]; community reports of the actual runtime
    output disagree on whether these come back plain ("angry") or
    bilingual ("生气/angry") depending on model/FunASR version, and this
    couldn't be verified directly here (funasr isn't installed in this
    environment). Handle both: take the part after the last "/" if
    present, else the label as-is.
    """
    return label.rsplit("/", 1)[-1].strip()


class Emotion2vec_emotion(Featureset):
    """Emotion2vec's own classification scores (angry/happy/neutral/...),
    for nkululeko.autopredict's `--model emotion` target."""

    def __init__(self, name, data_df, feats_type):
        super().__init__(name, data_df, feats_type)
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        self.model_initialized = False
        self.feats_type = feats_type

    def init_model(self):
        self.util.debug("loading emotion2vec classification model...")
        try:
            from funasr import AutoModel
        except ImportError:
            self.util.error(
                "FunASR is required for emotion2vec emotion prediction. "
                "Please install with: pip install funasr"
            )

        model_path = self.util.config_val(
            "FEATS",
            "emotion2vec.model",
            _MODEL_MAPPING.get(self.feats_type, "emotion2vec/emotion2vec_plus_large"),
        )
        try:
            self.model = AutoModel(model=model_path, hub="hf")
            self.util.debug(
                f"initialized emotion2vec classification model: {model_path}"
            )
            self.model_initialized = True
        except Exception as e:
            self.util.error(f"Failed to load emotion2vec model: {str(e)}")

    def extract(self):
        """Predict emotion classes, or re-open them when found on disk."""
        store = self.util.get_path("store")
        store_format = self.util.config_val("FEATS", "store_format", "pkl")
        storage = f"{store}{self.name}.{store_format}"
        if self._needs_extraction(storage):
            if not self.model_initialized:
                self.init_model()
            self.util.debug(
                "predicting emotion2vec emotion classes, this might take a while..."
            )
            rows = []
            for file, start, end in tqdm(self.data_df.index.to_list()):
                rows.append(self._predict_one(file, start, end))
            self.df = pd.DataFrame(rows, index=self.data_df.index).fillna(0.0)
            self.util.write_store(self.df, storage, store_format)
            try:
                self.context.config["DATA"]["needs_feature_extraction"] = "false"
            except KeyError:
                pass
        else:
            self.util.debug(f"reusing extracted emotion2vec classes: {storage}.")
            self.df = self.util.get_store(storage, store_format)

    def _predict_one(self, file, start, end):
        audio_path, is_temp = self._segment_to_wav(file, start, end)
        try:
            res = self.model.generate(
                audio_path, granularity="utterance", extract_embedding=False
            )
        except Exception as e:
            self.util.warn(f"emotion2vec prediction failed for {file}: {e}")
            return {}
        finally:
            if is_temp:
                os.unlink(audio_path)
        return self._scores_from_result(res, file)

    def _scores_from_result(self, res, file):
        if not isinstance(res, list) or not res:
            self.util.warn(f"no result from emotion2vec model for file: {file}")
            return {}
        labels = res[0].get("labels", [])
        scores = res[0].get("scores", [])
        return {_clean_label(label): score for label, score in zip(labels, scores)}

    def _segment_to_wav(self, file, start, end):
        """Return (path, is_temp): a 16 kHz mono wav holding just the
        (start, end) segment of `file` in a new temp file (is_temp=True,
        caller must remove it), or `file` itself unchanged when start/end
        don't look like a real sub-segment."""
        if not (hasattr(start, "total_seconds") and hasattr(end, "total_seconds")):
            return file, False

        import soundfile as sf

        # Read only the header for the sample rate -- torchaudio.load()
        # would decode the whole file just to discover this, doubling
        # I/O and memory per file (autopredict commonly processes full
        # files through this path).
        sampling_rate = sf.info(file).samplerate
        start_sample = int(start.total_seconds() * sampling_rate)
        num_samples = int((end - start).total_seconds() * sampling_rate)
        signal, sampling_rate = torchaudio.load(
            file, frame_offset=start_sample, num_frames=num_samples
        )
        if sampling_rate != 16000:
            signal = torchaudio.transforms.Resample(sampling_rate, 16000)(signal)
            sampling_rate = 16000

        signal_np = signal.squeeze().numpy()
        if signal_np.ndim > 1:
            signal_np = signal_np[0]

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(tmp_path, signal_np, sampling_rate)
        return tmp_path, True

    def extract_sample(self, signal, sr):
        if not self.model_initialized:
            self.init_model()

        import audiofile

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            audiofile.write(tmp_path, signal, sr)
            res = self.model.generate(
                tmp_path, granularity="utterance", extract_embedding=False
            )
        finally:
            os.remove(tmp_path)
        scores = self._scores_from_result(res, "<sample>")
        return np.array(list(scores.values()))
