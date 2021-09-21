# plots.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import ast
from util import Util
import glob_conf
import seaborn as sns
import numpy as np

class Plots():
    
    def __init__(self):
        """Initializing the util system"""
        self.util = Util()

    def describe_df(self, name, df, target, filename):
        """Make a stacked barplot of samples and speakers per sex and target values. speaker, gender and target columns must be present"""
        fig_dir = self.util.get_path('fig_dir')+'../' # one up because of the runs 
        sampl_num = df.shape[0]
        spkr_num = df.speaker.nunique()
        female_smpl_num = df[df.gender=='female'].shape[0]
        male_smpl_num = df[df.gender=='male'].shape[0]
        self.util.debug(f'{name}: # samples: {sampl_num} (f: {female_smpl_num}, m: {male_smpl_num}), # speakers: {spkr_num}')
        fig, axes = plt.subplots(nrows=1, ncols=2)
        df.groupby(target)['gender'].value_counts().unstack().plot(kind='bar', stacked=True, ax=axes[0], title=f'samples ({sampl_num})')
        df.groupby(target)['speaker'].nunique().plot(kind='bar', ax=axes[1], title=f'speakers ({spkr_num})')
        plt.tight_layout()
        plt.savefig(fig_dir+filename)
        fig.clear()
        plt.close(fig)

    def plotTsne(self, feats, labels, filename, perplexity=30, learning_rate=200):
        """Make a TSNE plot to see whether features are useful for classification"""
        fig_dir = self.util.get_path('fig_dir')+'../' # one up because of the runs 
        filename = fig_dir+filename
        self.util.debug(f'plotting tsne to {filename}, this might take a while...')
        model = TSNE(n_components=2, random_state=0, perplexity=perplexity, learning_rate=learning_rate)
        tsne_data = model.fit_transform(feats)
        tsne_data_labs = np.vstack((tsne_data.T, labels)).T
        tsne_df = pd.DataFrame(data=tsne_data_labs, columns=('Dim_1', 'Dim_2', 'label'))
        fg = sns.FacetGrid(tsne_df, hue='label', size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
        fig = fg.fig
        plt.tight_layout()
        plt.savefig(filename)
        fig.clear()
        plt.close(fig)
