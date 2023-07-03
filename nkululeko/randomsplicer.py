# randomsplicer.py

"""
Code originally by Oliver Pauly

Based on an idea by Klaus Scherer

K. R. Scherer, “Randomized splicing: A note on a simple technique for masking speech content” 
Journal of Experimental Research in Personality, vol. 5, pp. 155–159, 1971.

Evaluated in:
F. Burkhardt, Anna Derington, Matthias Kahlau, Klaus Scherer, Florian Eyben and Björn Schuller: Masking Speech Contents by Random Splicing: is Emotional Expression Preserved?, Proc. ICASSP, 2023

"""

import pandas as pd
from nkululeko.util import Util 
import nkululeko.randomsplicing as rsp
import numpy as np
import audiofile as af
import os
from audformat.utils import map_file_path
import audeer

class Randomsplicer:
    """
        augmenting the train split
    """
    def __init__(self, df):
        self.df = df
        self.util = Util('randomsplicer')

    def changepath(self, fp, np):
        fullpath = os.path.dirname(fp)
        return fp.replace(fullpath, np)

    def run(self, sample_selection):
        """
        random splice the selected samples and return a dataframe with new files index.
            adjustable parameters:
            * p_reverse: probability of some samples to be in reverse order (default: 0.3)
            * top_db: top db level for silence to be recognized (default: 12)   
        """

        p_reverse=0.3
        top_db=12

        files = self.df.index.get_level_values(0).values
        store = self.util.get_path('store')
        filepath = f'{store}randomspliced/'
        audeer.mkdir(filepath)
        self.util.debug(f'random splicing {sample_selection} samples to {filepath}')
        newpath = ''
        for i, f in enumerate(files):
            signal, sr = af.read(f)
            filename = os.path.basename(f)
            parent = os.path.dirname(f).split('/')[-1]
            sig_new = rsp.random_splicing(
                signal, sr,
                p_reverse=p_reverse,
                top_db=top_db,
                )

            newpath = f'{filepath}/{parent}/'
            audeer.mkdir(newpath)
            af.write(f'{newpath}{filename}', signal=sig_new, sampling_rate=sr)
            if i%10==0:
                print(f'random spliced {i} of {len(files)}')
        df_ret = self.df.copy()
        df_ret = df_ret.set_index(map_file_path(df_ret.index, lambda x: self.changepath(x, newpath)))
        db_filename = self.util.config_val('DATA', 'random_splice_result', 'random_spliced.csv')
        target = self.util.config_val('DATA', 'target', 'emotion')
        df_ret[target] = df_ret['class_label']
        df_ret = df_ret.drop(columns=['class_label'])
        df_ret.to_csv(db_filename)
        return df_ret