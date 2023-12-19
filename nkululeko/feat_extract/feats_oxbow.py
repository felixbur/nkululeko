# feats_oxbow.py

from nkululeko.utils.util import Util
from nkululeko.feat_extract.featureset import Featureset
import os
import pandas as pd
import opensmile


class Openxbow(Featureset):
    """Class to extract openXBOW processed opensmile features (https://github.com/openXBOW)"""

    def __init__(self, name, data_df, is_train=False):
        """Constructor. is_train is needed to distinguish from test/dev sets, because they use the codebook from the training"""
        super().__init__(name, data_df)
        self.is_train = is_train

    def extract(self):
        """Extract the features or load them from disk if present."""
        self.featset = self.util.config_val("FEATS", "set", "eGeMAPSv02")
        self.feature_set = eval(f"opensmile.FeatureSet.{self.featset}")
        store = self.util.get_path("store")
        storage = f"{store}{self.name}_{self.featset}.pkl"
        extract = self.util.config_val("FEATS", "needs_feature_extraction", False)
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        if extract or no_reuse or not os.path.isfile(storage):
            # extract smile features first
            self.util.debug("extracting openSmile features, this might take a while...")
            smile = opensmile.Smile(
                feature_set=self.feature_set,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                num_workers=5,
            )
            if isinstance(self.data_df.index, pd.MultiIndex):
                is_multi_index = True
                smile_df = smile.process_index(self.data_df.index)
            else:
                smile_df = smile.process_files(self.data_df.index)
            smile_df.index = smile_df.index.droplevel(1)
            smile_df.index = smile_df.index.droplevel(1)
            # compute xbow features
            # set some file names on disk
            lld_name, xbow_name, codebook_name = (
                "llds.csv",
                "xbow.csv",
                "xbow_codebook",
            )
            # save the smile features
            smile_df.to_csv(lld_name, sep=";", header=False)
            # get the path of the xbow java jar file
            xbow_path = self.util.config_val("FEATS", "xbow.model", "../openXBOW/")
            # get the size of the codebook
            size = self.util.config_val("FEATS", "size", 500)
            # get the number of assignements
            assignments = self.util.config_val("FEATS", "assignments", 10)
            # differentiate between train and test
            if self.is_train:
                # store the codebook
                os.system(
                    f"java -jar {xbow_path}openXBOW.jar -i"
                    f" {lld_name} -standardizeInput -log                     -o"
                    f" {xbow_name} -size {size} -a {assignments} -B"
                    f" {codebook_name}"
                )
            else:
                # use the codebook
                os.system(
                    f"java -jar {xbow_path}openXBOW.jar -i {lld_name}          "
                    f"           -o {xbow_name} -b {codebook_name}"
                )
            # read in the result from disk
            xbow_df = pd.read_csv(xbow_name, sep=";", header=None)
            # set the index
            xbow_df = xbow_df.set_index(self.data_df.index)
            # check if smile features should be added
            with_os = self.util.config_val("FEATS", "with_os", False)
            if with_os:
                # extract smile functionals
                self.util.debug(
                    "extracting openSmile functionals, this might take a" " while..."
                )
                smile = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,  # always use eGemaps for this
                    feature_level=opensmile.FeatureLevel.Functionals,
                    num_workers=5,
                )
                if isinstance(self.data_df.index, pd.MultiIndex):
                    is_multi_index = True
                    smile_df = smile.process_index(self.data_df.index)
                else:
                    smile_df = smile.process_files(self.data_df.index)
                # drop the multi index
                smile_df.index = smile_df.index.droplevel(1)
                smile_df.index = smile_df.index.droplevel(1)
                xbow_df = xbow_df.join(smile_df)
            # in any case, store to disk for later use
            xbow_df.to_pickle(storage)
            # and assign to be the "official" feature set
            self.df = xbow_df
        else:
            self.util.debug("reusing extracted OS features.")
            self.df = pd.read_pickle(storage)
