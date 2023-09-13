# dataset_csv.py
import os
import os.path

import audformat.utils
import pandas as pd
from nkululeko.data.dataset import Dataset


class Dataset_CSV(Dataset):
    """Class to represent datasets stored as a csv file"""

    def load(self):
        """Load the dataframe with files, speakers and task labels"""
        self.util.debug(f"loading {self.name}")
        self.got_target, self.got_speaker, self.got_gender = False, False, False
        data_file = self.util.config_val_data(self.name, "", "")
        if not os.path.isabs(data_file):
            exp_root = self.util.config_val("EXP", "root", "")
            data_file = os.path.join(exp_root, data_file)
        root = os.path.dirname(data_file)
        audio_path = self.util.config_val_data(self.name, "audio_path", "")
        df = audformat.utils.read_csv(data_file)
        absolute_path = eval(
            self.util.config_val_data(self.name, "absolute_path", True)
        )
        if not absolute_path:
            # add the root folder to the relative paths of the files
            if audformat.index_type(df.index) == "segmented":
                file_index = (
                    df.index.levels[0]
                    .map(lambda x: root + "/" + audio_path + "/" + x)
                    .values
                )
                df = df.set_index(df.index.set_levels(file_index, level="file"))
            else:
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                df = df.set_index(
                    df.index.to_series().apply(
                        lambda x: root + "/" + audio_path + "/" + x
                    )
                )
        self.df = df
        self.db = None
        self.got_target = True
        self.is_labeled = self.got_target
        self.start_fresh = eval(self.util.config_val("DATA", "no_reuse", "False"))
        if self.is_labeled and not "class_label" in self.df.columns:
            self.df["class_label"] = self.df[self.target]
        if "gender" in self.df.columns:
            self.got_gender = True
        elif "sex" in self.df.columns:
            self.df = self.df.rename(columns={'sex':'gender'})            
            self.got_gender = True
        if "age" in self.df.columns:
            self.got_age = True
        if "speaker" in self.df.columns:
            self.got_speaker = True
        speaker_num = 0
        if self.got_speaker:
            speaker_num = self.df.speaker.nunique()
        self.util.debug(
            f"Loaded database {self.name} with {df.shape[0]} "
            f"samples: got targets: {self.got_target}, got speakers: {self.got_speaker} ({speaker_num}), "
            f"got sexes: {self.got_gender}, "
            f"got age: {self.got_age}"
        )

    def prepare(self):
        super().prepare()
