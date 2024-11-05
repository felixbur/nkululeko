# dataset_csv.py
import ast
import os
import os.path

import audformat.utils
import pandas as pd

import nkululeko.glob_conf as glob_conf
from nkululeko.data.dataset import Dataset
from nkululeko.reporting.report_item import ReportItem


class Dataset_CSV(Dataset):
    """Class to represent datasets stored as a csv file"""

    def load(self):
        """Load the dataframe with files, speakers and task labels"""
        self.util.debug(f"loading {self.name}")
        self.got_target, self.got_speaker, self.got_gender = False, False, False
        data_file = self.util.config_val_data(self.name, "", "")
        # if not os.path.isabs(data_file):
        #     exp_root = self.util.config_val("EXP", "root", "")
        #     data_file = os.path.join(exp_root, data_file)
        root = os.path.dirname(data_file)
        audio_path = self.util.config_val_data(self.name, "audio_path", "./")
        df = pd.read_csv(data_file)
        # trim all string values
        df_obj = df.select_dtypes("object")
        df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
        # special treatment for segmented dataframes with only one column:
        if "start" in df.columns and len(df.columns) == 4:
            index = audformat.segmented_index(
                df.file.values, df.start.values, df.end.values
            )
            df = df.set_index(index)
            df = df.drop(columns=["file", "start", "end"])
        else:
            df = audformat.utils.read_csv(data_file)
        if isinstance(df, pd.Series):
            df = df.to_frame()
        elif isinstance(df, pd.Index):
            df = pd.DataFrame(index=df)
        rename_cols = self.util.config_val_data(self.name, "colnames", False)
        if rename_cols:
            col_dict = ast.literal_eval(rename_cols)
            df = df.rename(columns=col_dict)
        absolute_path = eval(
            self.util.config_val_data(self.name, "absolute_path", "True")
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
        else:  # absolute path is True
            if audformat.index_type(df.index) == "segmented":
                file_index = (
                    df.index.levels[0].map(lambda x: audio_path + "/" + x).values
                )
                df = df.set_index(df.index.set_levels(file_index, level="file"))
            else:
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                df = df.set_index(
                    df.index.to_series().apply(lambda x: audio_path + "/" + x)
                )

        self.df = df
        self.db = None
        target = self.util.config_val("DATA", "target", None)
        if target is not None:
            self.got_target = True
        else:
            self.got_target = False
        self.is_labeled = self.got_target
        self.start_fresh = eval(self.util.config_val("DATA", "no_reuse", "False"))
        is_index = False
        try:
            if self.is_labeled and "class_label" not in self.df.columns:
                self.df["class_label"] = self.df[self.target]
        except AttributeError:
            is_index = True
            r_string = (
                f"Loaded database {self.name} with {df.shape[0]} samples as index."
            )
        if not is_index:
            if "gender" in self.df.columns:
                self.got_gender = True
            if "age" in self.df.columns:
                self.got_age = True
            if "speaker" in self.df.columns:
                self.got_speaker = True
            speaker_num = 0
            if self.got_speaker:
                speaker_num = self.df.speaker.nunique()
            r_string = (
                f"Loaded database {self.name} with {df.shape[0]} samples: got"
                f" targets: {self.got_target}, got speakers:"
                f" {self.got_speaker} ({speaker_num}), got sexes:"
                f" {self.got_gender}, got age: {self.got_age}"
            )
        self.util.debug(r_string)
        glob_conf.report.add_item(ReportItem("Data", "Loaded report", r_string))

    def prepare(self):
        super().prepare()
