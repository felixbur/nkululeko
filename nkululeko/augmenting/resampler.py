"""
resample a data frame

"""
import os
import shutil

import audformat
import pandas as pd
import torchaudio
from nkululeko.utils.util import Util


class Resampler:
    def __init__(self, df, not_testing=True):
        self.SAMPLING_RATE = 16000
        self.df = df
        self.util = Util("resampler", has_config=not_testing)
        self.util.warn(f"all files might be resampled to {self.SAMPLING_RATE}")
        self.not_testing = not_testing

    def resample(self):
        files = self.df.index.get_level_values(0).values
        replace = eval(self.util.config_val("RESAMPLE", "replace", "False"))
        if self.not_testing:
            store = self.util.get_path("store")
        else:
            store = "./"
        tmp_audio = "tmp_resample.wav"
        succes, error = 0, 0
        if not replace:
            new_files = []
        for i, f in enumerate(files):
            signal, org_sr = torchaudio.load(f"{f}")  # handle spaces
            # convert to mono if stereo
            if signal.shape[0] > 1:
                signal = signal.mean(dim=0, keepdim=True)
            # if f cannot be loaded, give warning and skip
            if signal.shape[0] == 0:
                self.util.warn(f"cannot load {f}")
                error += 1
                continue
            if org_sr != self.SAMPLING_RATE:
                self.util.debug(f"resampling {f} (sr = {org_sr})")
                resampler = torchaudio.transforms.Resample(org_sr, self.SAMPLING_RATE)
                signal = resampler(signal)
                if replace:
                    torchaudio.save(
                        os.path.splitext(f)[0] + ".wav",
                        signal,
                        self.SAMPLING_RATE,
                    )
                else:
                    new_file_name = os.path.splitext(f)[0] + "_16kHz.wav"
                    torchaudio.save(new_file_name, signal, self.SAMPLING_RATE)
                    new_files.append(new_file_name)
                succes += 1
        if not replace:
            self.df = self.df.set_index(
                self.df.index.set_levels(new_files, level="file")
            )
            target_file = self.util.config_val("RESAMPLE", "target", "resampled.csv")
            # remove encoded labels
            target = self.util.config_val("DATA", "target", "emotion")
            if "class_label" in self.df.columns:
                self.df = self.df.drop(columns=[target])
                self.df = self.df.rename(columns={"class_label": target})
            # save file
            self.df.to_csv(target_file)
            self.util.debug(
                "saved resampled list of files to" f" {os.path.abspath(target_file)}"
            )
        self.util.debug(f"resampled {succes} files, {error} errors")


def main():
    testfile = "test_wavs/audio441.wav"
    shutil.copyfile(testfile, "tmp.wav")
    files = pd.Series([testfile])
    df_sample = pd.DataFrame(index=files)
    df_sample["target"] = "anger"
    df_sample.index = audformat.utils.to_segmented_index(
        df_sample.index, allow_nat=False
    )
    df_sample.head(10)
    resampler = Resampler(df_sample, not_testing=False)
    resampler.resample()
    shutil.copyfile(testfile, "tmp.resample_result.wav")
    shutil.copyfile("tmp.wav", testfile)


if __name__ == "__main__":
    main()
