# feats_ast.py
import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTForAudioClassification

import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.featureset import Featureset


class CustomASTFeatureExtractor(ASTFeatureExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_rate = 16000

    def __call__(self, *args, **kwargs):
        kwargs["sampling_rate"] = self.sampling_rate
        result = super().__call__(*args, **kwargs)
        result["pixel_values"] = result["input_values"]
        return result


class Ast(Featureset):
    """Class to extract AST (Audio Spectrogram Transformer) embeddings"""

    def __init__(self, name, data_df, feat_type):
        super().__init__(name, data_df, feat_type)
        cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.util.config_val("MODEL", "device", cuda)
        self.model_initialized = False
        self.feat_type = feat_type

    def init_model(self):
        self.util.debug("loading AST model...")
        model_path = self.util.config_val(
            "FEATS", "ast.model", "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self.feature_extractor = CustomASTFeatureExtractor.from_pretrained(model_path)
        self.model = ASTForAudioClassification.from_pretrained(model_path).to(
            self.device
        )
        print(f"initialized AST model on {self.device}")
        self.model.eval()
        self.model_initialized = True

    def extract(self):
        """Extract the features or load them from disk if present."""
        if self.data_df is None:
            self.util.error("data_df is None. Make sure it's properly initialized.")
            return

        store = self.util.get_path("store")
        storage = f"{store}{self.name}.pkl"
        extract = self.util.config_val("FEATS", "needs_feature_extraction", False)
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))

        if extract or no_reuse or not os.path.isfile(storage):
            if not self.model_initialized:
                self.init_model()
            self.util.debug("extracting AST embeddings, this might take a while...")
            emb_series = []
            for idx, row in tqdm(self.data_df.iterrows(), total=len(self.data_df)):
                try:
                    file, start, end = (
                        row.name
                        if isinstance(row.name, tuple)
                        else (row.name, None, None)
                    )

                    signal, sampling_rate = torchaudio.load(file)
                    if start is not None and end is not None:
                        start_frame = int(start.total_seconds() * sampling_rate)
                        end_frame = int(end.total_seconds() * sampling_rate)
                        signal = signal[:, start_frame:end_frame]

                    if sampling_rate != 16000:
                        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
                        signal = resampler(signal)
                        sampling_rate = 16000

                    emb = self.get_embeddings(signal, sampling_rate, file)
                    emb_series.append(emb)
                except Exception as e:
                    self.util.error(f"Error processing file {file}: {str(e)}")
                    emb_series.append(
                        np.zeros(self.model.config.hidden_size)
                    )  # Append zero vector on error

            self.df = pd.DataFrame(emb_series, index=self.data_df.index)
            self.df.to_pickle(storage)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "false"
            except KeyError:
                pass
        else:
            self.util.debug("reusing extracted AST embeddings")
            self.df = pd.read_pickle(storage)

        if self.df.isnull().values.any():
            nanrows = self.df.index[self.df.isnull().any(axis=1)].tolist()
            self.util.error(f"got nan: {self.df.shape} {self.df.isnull().sum().sum()}")
            self.util.error(f"Rows with NaN: {nanrows}")

    def get_embeddings(self, signal, sampling_rate, file):
        """Extract embeddings from raw audio signal."""
        try:
            with torch.no_grad():
                # Check if the audio is long enough
                min_length = 400  # Minimum length required by the model
                if signal.shape[1] < min_length:
                    # If audio is too short, repeat it until it reaches the minimum length
                    repeat_times = int(np.ceil(min_length / signal.shape[1]))
                    signal = signal.repeat(1, repeat_times)
                    signal = signal[
                        :, :min_length
                    ]  # Trim to exact length if it went over

                inputs = self.feature_extractor(
                    signal, sampling_rate=sampling_rate, return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get the hidden states
                outputs = self.model(**inputs, output_hidden_states=True)

                # Get the hidden states from the last layer
                last_hidden_state = outputs.hidden_states[-1]

                # Average pooling over the time dimension
                embeddings = torch.mean(last_hidden_state, dim=1)

                return embeddings.squeeze().cpu().numpy()
        except Exception as e:
            self.util.error(f"Error extracting embeddings for file {file}: {str(e)}")
            return np.zeros(
                self.model.config.hidden_size
            )  # Return zero vector on error

    def extract_sample(self, signal, sr):
        self.init_model()
        feats = self.get_embeddings(signal, sr, "no file")
        return feats
