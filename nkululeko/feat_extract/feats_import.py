# feats_import.py

import os
import ast
import audformat
import pandas as pd
from nkululeko.utils.util import Util
from nkululeko.feat_extract.featureset import Featureset


class Importset(Featureset):
    """Class to import features that have been compiled elsewhere"""

    def __init__(self, name, data_df):
        super().__init__(name, data_df)

    def extract(self):
        """Import the features."""
        self.util.debug(f"importing features for {self.name}")
        try:
            feat_import_files = self.util.config_val("FEATS", "import_file", False)
            feat_import_files = ast.literal_eval(feat_import_files)
        except ValueError as e:
            self.util.error(
                "feature type == import needs import_file = ['file1', 'filex']"
            )
        except SyntaxError as se:
            if type(feat_import_files) == str:
                feat_import_files = [feat_import_files]
            else:
                self.util.error(f"import_file is wrong: {feat_import_files}")

        feat_df = pd.DataFrame()
        for feat_import_file in feat_import_files:
            if not os.path.isfile(feat_import_file):
                self.util.error(f"no import file: {feat_import_file}")
            df = audformat.utils.read_csv(feat_import_file)
            df = df[df.index.isin(self.data_df.index)]
            feat_df = pd.concat([feat_df, df])
        if feat_df.shape[0] == 0:
            self.util.error(f"Imported features for data set {self.name} not found!")
        # and assign to be the "official" feature set
        self.df = feat_df
