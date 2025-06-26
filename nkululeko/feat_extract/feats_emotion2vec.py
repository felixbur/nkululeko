# feats_emotion2vec.py
# emotion2vec feature extractor for Nkululeko
# choices for feat_type = "emotion2vec", "emotion2vec-large", "emotion2vec-base", "emotion2vec-seed"

# requirements:
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
                "Please install with: pip install funasr"
            )

        # Map feat_type to model names on HuggingFace
        model_mapping = {
            "emotion2vec": "emotion2vec/emotion2vec_base",
            "emotion2vec-base": "emotion2vec/emotion2vec_base",
            "emotion2vec-seed": "emotion2vec/emotion2vec_plus_seed",
            "emotion2vec-large": "emotion2vec/emotion2vec_plus_large",
        }

        # Get model path from config or use default mapping
        model_path = self.util.config_val(
            "FEATS",
            "emotion2vec.model",
            model_mapping.get(self.feat_type, "emotion2vec/emotion2vec_base"),
        )

        try:
            # Initialize the FunASR model for emotion2vec using HuggingFace Hub
            self.model = AutoModel(
                model=model_path,
                hub="hf"  # Use HuggingFace Hub instead of ModelScope
            )
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
                emb = self.extract_embedding(file, start, end)
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

    def extract_embedding(self, file, start, end):
        """Extract embeddings directly from audio file."""
        try:
            # Handle segment extraction if needed
            if hasattr(start, "total_seconds") and hasattr(end, "total_seconds"):
                # Load audio segment
                _, sampling_rate = torchaudio.load(file, frame_offset=0, num_frames=-1)
                start_sample = int(start.total_seconds() * sampling_rate)
                num_samples = int((end - start).total_seconds() * sampling_rate)
                signal, sampling_rate = torchaudio.load(
                    file, frame_offset=start_sample, num_frames=num_samples
                )

                # Resample to 16kHz if needed
                if sampling_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
                    signal = resampler(signal)
                    sampling_rate = 16000

                # Convert to numpy and save as temporary file
                signal_np = signal.squeeze().numpy()
                if signal_np.ndim > 1:
                    signal_np = signal_np[0]  # Take first channel if stereo

                import tempfile
                import soundfile as sf

                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp_file:
                    sf.write(tmp_file.name, signal_np, sampling_rate)
                    audio_path = tmp_file.name
            else:
                # Use full file directly
                audio_path = file

            try:
                # Extract features using FunASR emotion2vec model
                res = self.model.generate(
                    audio_path, granularity="utterance", extract_embedding=True
                )

                # Get the embeddings from the result
                if isinstance(res, list) and len(res) > 0:
                    embeddings = res[0].get("feats", None)
                    if embeddings is not None:
                        if isinstance(embeddings, list):
                            embeddings = np.array(embeddings)
                        return embeddings.flatten()
                    else:
                        # Fallback based on model type
                        if 'large' in self.feat_type.lower():
                            return np.array([0.0] * 1024)
                        else:
                            return np.array([0.0] * 768)
                else:
                    self.util.error(
                        f"No result from emotion2vec model for file: {file}"
                    )
                    # Fallback based on model type
                    if 'large' in self.feat_type.lower():
                        return np.array([0.0] * 1024)
                    else:
                        return np.array([0.0] * 768)

            finally:
                # Clean up temporary file if we created one
                if hasattr(start, "total_seconds") and hasattr(end, "total_seconds"):
                    os.unlink(audio_path)

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            self.util.error(f"couldn't extract file: {file}, error: {str(e)}")
            # Return appropriate dimension based on model type
            if 'large' in self.feat_type.lower():
                return np.array([0.0] * 1024)
            else:
                return np.array([0.0] * 768)

    def extract_sample(self, signal, sr):
        """Extract features from a single sample."""
        if not self.model_initialized:
            self.init_model()

        # Save signal as temporary file for emotion2vec
        import tempfile
        import soundfile as sf

        try:
            # Convert tensor to numpy if needed
            if torch.is_tensor(signal):
                signal_np = signal.squeeze().numpy()
            else:
                signal_np = signal.squeeze()

            # Handle multi-channel audio
            if signal_np.ndim > 1:
                signal_np = signal_np[0]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, signal_np, sr)

                # Extract using the emotion2vec model
                res = self.model.generate(
                    tmp_file.name, granularity="utterance", extract_embedding=True
                )

                # Get embeddings from result
                if isinstance(res, list) and len(res) > 0:
                    embeddings = res[0].get("feats", None)
                    if embeddings is not None:
                        if isinstance(embeddings, list):
                            embeddings = np.array(embeddings)
                        return embeddings.flatten()

                # Fallback based on model type
                if 'large' in self.feat_type.lower():
                    return np.array([0.0] * 1024)
                else:
                    return np.array([0.0] * 768)

        except Exception as e:
            print(f"Error in extract_sample: {str(e)}")
            # Return appropriate dimension based on model type
            if 'large' in self.feat_type.lower():
                return np.array([0.0] * 1024)
            else:
                return np.array([0.0] * 768)
        finally:
            # Clean up temporary file
            if tmp_file is not None:  # Check if tmp_file was created
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass
