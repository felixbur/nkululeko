import pandas as pd
from nkululeko.utils.util import Util
import os


class FileChecker:
    def __init__(self, df):
        self.util = Util("filechecker")
        self.df = df.copy()
        self.util.copy_flags(df, self.df)
        check_vad = self.util.config_val("DATA", "check_vad", False)
        if check_vad:
            self.util.debug(f"This may take a while downloading the VAD model")
            import torch

            torch.set_num_threads(1)
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )

    def set_data(self, df):
        self.df = df.copy()
        self.util.copy_flags(df, self.df)

    def all_checks(self, data_name=""):
        self.check_size(data_name)
        return self.check_vad(data_name)

    def check_size(self, data_name=""):
        """limit number of samples
        the samples are selected randomly
        """
        if data_name == "":
            min = self.util.config_val("DATA", "check_size", False)
        else:
            min = self.util.config_val_data(data_name, "check_size", False)
        if min:
            if min == "True":
                min = (
                    1000  # 1000 bytes would be a reasonable minimal size for 16 kHz sr
                )
            old_samples = self.df.shape[0]
            df = self.df.copy()
            for i in self.df.index:
                file = i[0]
                file_size = os.path.getsize(file)
                if file_size < int(min):
                    df = df.drop(i, axis=0)
            self.util.debug(
                f"{data_name}: checked for samples less than {min} bytes,"
                f" reduced samples from {old_samples} to {df.shape[0]}"
            )
            self.util.copy_flags(self.df, df)
            self.df = df
            return df
        else:
            return self.df

    def check_vad(self, data_name=""):
        """limit number of samples
        the samples are selected randomly
        """
        if data_name == "":
            check = self.util.config_val("DATA", "check_vad", False)
        else:
            check = self.util.config_val_data(data_name, "check_vad", False)
        if check:
            self.util.debug(f"{data_name}: checking for samples without speech.")
            SAMPLING_RATE = 16000
            (
                get_speech_timestamps,
                save_audio,
                read_audio,
                VADIterator,
                collect_chunks,
            ) = self.vad_utils
            old_samples = self.df.shape[0]
            df = self.df.copy()
            for i in self.df.index:
                file = i[0]
                wav = read_audio(file, sampling_rate=SAMPLING_RATE)
                speech_timestamps = get_speech_timestamps(
                    wav, self.vad_model, sampling_rate=SAMPLING_RATE
                )
                if len(speech_timestamps) == 0:
                    self.util.debug(f"{file}: no speech detected")
                    df = df.drop(i, axis=0)
            self.util.debug(
                f"{data_name}: checked for samples without speech, reduced"
                f" samples from {old_samples} to {df.shape[0]}"
            )
            self.util.copy_flags(self.df, df)
            self.df = df
            return df
        else:
            return self.df
