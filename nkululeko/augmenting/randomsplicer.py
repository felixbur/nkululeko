# randomsplicer.py

"""
Code originally by Oliver Pauly

Based on an idea by Klaus Scherer

K. R. Scherer, “Randomized splicing: A note on a simple technique for masking speech content” 
Journal of Experimental Research in Personality, vol. 5, pp. 155–159, 1971.

Evaluated in:
F. Burkhardt, Anna Derington, Matthias Kahlau, Klaus Scherer, Florian Eyben and Björn Schuller: Masking Speech Contents by Random Splicing: is Emotional Expression Preserved?, Proc. ICASSP, 2023

"""

import os

import audeer
import audiofile as af
import pandas as pd
from tqdm import tqdm

import nkululeko.augmenting.randomsplicing as rsp
from nkululeko.utils.util import Util


class Randomsplicer:
    """
    augmenting the train split
    """

    def __init__(self, df):
        self.df = df
        self.util = Util("randomsplicer")

    def run(self, sample_selection):
        """
        random splice the selected samples and return a dataframe with new files index.
            adjustable parameters:
            * p_reverse: probability of some samples to be in reverse order (default: 0.3)
            * top_db: top db level for silence to be recognized (default: 12)
        """

        p_reverse = float(self.util.config_val("AUGMENT", "p_reverse", "0.3"))
        top_db = float(self.util.config_val("AUGMENT", "top_db", "12"))

        files = self.df.index.get_level_values(0).values
        store = self.util.get_path("store")
        filepath = f"{store}randomspliced/"
        audeer.mkdir(filepath)
        self.util.debug(
            f"random splicing {sample_selection} samples to {filepath}, "
            + f"p_reverse = {p_reverse}, top_db = {top_db}",
        )
        newpath = ""
        index_map = {}
        for i, f in enumerate(tqdm(files)):
            signal, sr = af.read(f)
            filename = os.path.basename(f)
            parent = os.path.dirname(f).split("/")[-1]
            sig_new = rsp.random_splicing(
                signal,
                sr,
                p_reverse=p_reverse,
                top_db=top_db,
            )
            newpath = f"{filepath}/{parent}/"
            audeer.mkdir(newpath)
            new_full_name = newpath + filename
            af.write(new_full_name, signal=sig_new, sampling_rate=sr)
            index_map[f] = new_full_name

        df_ret = self.df.copy()

        file_index = df_ret.index.to_series().map(lambda x: index_map[x[0]]).values
        # workaround because i just couldn't get this easier...
        arrays = [
            file_index,
            list(df_ret.index.get_level_values(1)),
            list(df_ret.index.get_level_values(2)),
        ]
        new_index = pd.MultiIndex.from_arrays(arrays, names=("file", "start", "end"))
        df_ret = df_ret.set_index(new_index)

        return df_ret
