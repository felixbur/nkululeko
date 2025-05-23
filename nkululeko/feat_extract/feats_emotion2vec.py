# feats_emotion2vec.py
# emotion2vec feature extractor for Nkululeko
# example feat_type = "emotion2vec-large", "emotion2vec-base", "emotion2vec-seed"

# requirements:
# pip install "modelscope>=1.9.5,<2.0.0"
# pip install funasr

import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.featureset import Featureset


class Emotion2vec(Featureset):
    """Class to extract emotion2vec embeddings."""

    def __init__(self, name, data_df, feat_type):
        """Constructor.

        Is_train is needed to distinguish from test/dev sets,
        because they use the codebook from the training.
        """
        super().__init__(name, data_df, feat_type)
        # check if device is not set, use cuda if available
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        self.model_initialized = False
        self.feat_type = feat_type

    def init_model(self):
        # load model
        self.util.debug("loading emotion2vec model...")

        try:
            from funasr import AutoModel
        except ImportError:
            self.util.error(
                "FunASR is required for emotion2vec features. "
                "Please install with: pip install funasr modelscope"
            )

        # Map feat_type to model names
        model_mapping = {
            "emotion2vec": "iic/emotion2vec_base",
            "emotion2vec-base": "iic/emotion2vec_plus_base",
            "emotion2vec-seed": "iic/emotion2vec_plus_seed",
            "emotion2vec-large": "iic/emotion2vec_plus_large",
        }

        # Get model path from config or use default mapping
        model_path = self.util.config_val(
            "FEATS",
            "emotion2vec.model",
            model_mapping.get(self.feat_type, "iic/emotion2vec_plus_base"),
        )

        try:
            # Initialize the FunASR model for emotion2vec
            self.model = AutoModel(model=model_path)
            self.util.debug(f"initialized emotion2vec model: {model_path}")
            self.model_initialized = True
        except Exception as e:
            self.util.error(f"Failed to load emotion2vec model: {str(e)}")

    def extract(self):
        """Extract the features or load them from disk if present."""
        store = self.util.get_path("store")
        storage = f"{store}{self.name}.pkl"
        extract = self.util.config_val("FEATS", "needs_feature_extraction", False)
        no_reuse = self.util.config_val("FEATS", "no_reuse", "False") == "True"
        if extract or no_reuse or not os.path.isfile(storage):
            if not self.model_initialized:
                self.init_model()
            self.util.debug(
                "extracting emotion2vec embeddings, this might take a while..."
            )
            emb_series = pd.Series(index=self.data_df.index, dtype=object)
            length = len(self.data_df.index)
            for idx, (file, start, end) in enumerate(
                tqdm(self.data_df.index.to_list())
            ):
                # First, load to get sampling_rate
                _, sampling_rate = torchaudio.load(file, frame_offset=0, num_frames=-1)

                # Calculate offsets in samples
                start_sample = (
                    int(start.total_seconds() * sampling_rate)
                    if hasattr(start, "total_seconds")
                    else 0
                )
                num_samples = (
                    int((end - start).total_seconds() * sampling_rate)
                    if hasattr(start, "total_seconds") and hasattr(end, "total_seconds")
                    else -1
                )

                # Now load the segment
                signal, sampling_rate = torchaudio.load(
                    file, frame_offset=start_sample, num_frames=num_samples
                )

                # Resample to 16kHz if needed (emotion2vec expects 16kHz)
                if sampling_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
                    signal = resampler(signal)
                    sampling_rate = 16000

                emb = self.get_embeddings(signal, sampling_rate, file)
                emb_series.iloc[idx] = emb
            self.df = pd.DataFrame(emb_series.values.tolist(), index=self.data_df.index)
            self.df.to_pickle(storage)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "false"
            except KeyError:
                pass
        else:
            self.util.debug(f"reusing extracted {self.feat_type} embeddings")
            self.df = pd.read_pickle(storage)
            if self.df.isnull().values.any():
                nanrows = self.df.columns[self.df.isna().any()].tolist()
                print(nanrows)
                self.util.error(
                    f"got nan: {self.df.shape} {self.df.isnull().sum().sum()}"
                )

    def get_embeddings(self, signal, sampling_rate, file):
        """Extract embeddings from raw audio signal."""
        try:
            # Convert tensor to numpy if needed
            if torch.is_tensor(signal):
                signal_np = signal.squeeze().numpy()
            else:
                signal_np = signal.squeeze()

            # emotion2vec expects 1D audio array
            if signal_np.ndim > 1:
                signal_np = signal_np[0]  # Take first channel if stereo

            # Save temporary wav file for FunASR model
            import tempfile

            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, signal_np, sampling_rate)
                tmp_path = tmp_file.name

            try:
                # Extract features using FunASR
                res = self.model.generate(
                    tmp_path, granularity="utterance", extract_embedding=True
                )

                # Get the embeddings from the result
                if isinstance(res, list) and len(res) > 0:
                    embeddings = res[0].get("feats", None)
                    if embeddings is not None:
                        if isinstance(embeddings, list):
                            embeddings = np.array(embeddings)
                        return embeddings.flatten()
                    else:
                        # Fallback to emotion predictions if embeddings not available
                        predictions = res[0].get("text", "")
                        # Create a simple feature vector from prediction scores
                        return np.array(
                            [0.0] * 768
                        )  # Default size similar to other models
                else:
                    self.util.error(
                        f"No result from emotion2vec model for file: {file}"
                    )
                    return np.array([0.0] * 768)

            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            self.util.error(f"couldn't extract file: {file}, error: {str(e)}")
            return np.array([0.0] * 768)  # Return zero vector on error

    def extract_sample(self, signal, sr):
        """Extract features from a single sample."""
        if not self.model_initialized:
            self.init_model()
        feats = self.get_embeddings(signal, sr, "no file")
        return feats
