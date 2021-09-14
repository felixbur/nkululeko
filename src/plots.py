# plots.py
import pandas as pd
import matplotlib.pyplot as plt
import ast
from util import Util
import glob_conf

class Plots():
    
    def __init__(self):
        """Initializing the util system"""
        self.util = Util()

    def describe_df(self, df, target, filename):
        """Make a stacked barplot of samples and speakers per sex and target values. speaker, gender and target columns must be present"""
        fig_dir = self.util.get_path('fig_dir')
        print('# samples: {}, # speakers: {}'.format(df.shape[0], df.speaker.nunique()))
        fig, axes = plt.subplots(nrows=1, ncols=2)
        df.groupby(target)['gender'].value_counts().unstack().plot(kind='bar', stacked=True, ax=axes[0], title='samples')
        df.groupby(target)['speaker'].nunique().plot(kind='bar', ax=axes[1], title='speakers')
        plt.tight_layout()
        plt.savefig(fig_dir+filename)
        fig.clear()
        plt.close(fig)