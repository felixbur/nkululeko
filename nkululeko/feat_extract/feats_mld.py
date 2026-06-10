# mld_fset.py
import os
import sys
from typing import Optional

import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.featureset import Featureset


class MLD_set(Featureset):
    def __init__(
        self,
        name,
        data_df,
        feats_type: Optional[str] = None,
    ):
        super().__init__(name, data_df, feats_type)
        mld_path = self.util.config_val("FEATS", "mld.model", None)
        if mld_path is None:
            self.util.error("FEATS.mld.model required")
        sys.path.append(mld_path)

    def extract(self):
        """Extract mid-level descriptor features using the audmld package.

        The MLD class used for extraction is controlled by FEATS.mld.df (default: Mld).
        Accepted values: Mld, MldSust, MldStruct.
        Cache is keyed per class name so switching classes does not reuse stale features.
        """
        store = self.util.get_path("store")
        storage = f"{store}{self.name}.pkl"
        import audmld

        mld_class_name = self.util.config_val("FEATS", "mld.df", "Mld")
        mld_classes = {"Mld": audmld.Mld, "MldSust": audmld.MldSust, "MldStruct": audmld.MldStruct}
        if mld_class_name not in mld_classes:
            self.util.error(
                f"FEATS.mld.df must be one of {list(mld_classes.keys())}, got '{mld_class_name}'"
            )
        self.util.debug(f"using MLD class: {mld_class_name}")
        cache_root = f"{storage}_{mld_class_name}"
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        if no_reuse and os.path.exists(cache_root):
            os.remove(cache_root)
        if not os.path.isfile(cache_root):
            self.util.debug(
                "extracting midleveldescriptor features, this might take a while..."
            )
        else:
            self.util.debug("reusing previously extracted midleveldescriptor features")
        fex_mld = mld_classes[mld_class_name](num_workers=6, verbose=True)
        self.df = fex_mld.process_index(index=self.data_df.index, cache_root=cache_root)
        self.util.debug(f"MLD feats shape: {self.df.shape}")
        # shouldn't happen
        # replace NANa with column means values
        self.util.debug("MLD extractor: checking for NANs...")
        for col in self.df.columns:
            if self.df[col].isnull().values.any():
                self.util.debug(
                    f"{col} includes {self.df[col].isnull().sum()} nan,"
                    " inserting mean values"
                )
                self.df[col] = self.df[col].fillna(self.df[col].mean())

        try:
            # use only samples that have a minimum number of syllables
            min_syls = int(glob_conf.config["FEATS"]["min_syls"])
            self.df = self.df[self.df["hld_nSyl"] >= min_syls]
        except KeyError:
            pass
        if self.df.isna().to_numpy().any():
            self.util.error("feats 0: NANs exist")
        self.df = self.df.astype(float)
