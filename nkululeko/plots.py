# plots.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from nkululeko.util import Util
import seaborn as sns
import numpy as np
import ast

class Plots():
    
    def __init__(self):
        """Initializing the util system"""
        self.util = Util('plots')
        self.format = self.util.config_val('PLOT', 'format', 'png')
        self.target = self.util.config_val('DATA', 'target', 'emotion')

    def plot_distributions_speaker(self, df):
        df_speakers = pd.DataFrame()
        pd.options.mode.chained_assignment = None  # default='warn'
        for s in df.speaker.unique():
            df_speaker = df[df.speaker == s]
            df_speaker['samplenum'] = df_speaker.shape[0]
            df_speakers = pd.concat([df_speakers, df_speaker.head(1)])
        # plot the distrubution of samples per speaker        
        fig_dir = self.util.get_path('fig_dir')+'../' # one up because of the runs 
        self.util.debug(f'plotting samples per speaker')
        if 'gender' in df_speakers:
            filename = f'samples_value_counts'
            ax = df_speakers.groupby('samplenum')['gender'].value_counts().unstack().plot(kind='bar', stacked=True, title=f'samples per speaker ({df_speakers.shape[0]})', rot=0)
            ax.set_ylabel(f'number of speakers')
            ax.set_xlabel('number of samples')
            fig = ax.figure
            plt.tight_layout()
            plt.savefig(f'{fig_dir}{filename}.{self.format}')
            plt.close(fig)
            fig.clear()
        else:
            filename = f'samples_value_counts'
            ax = df_speakers['samplenum'].value_counts().sort_values().plot(kind='bar', stacked=True, title=f'samples per speaker ({df_speakers.shape[0]})', rot=0)
            ax.set_ylabel(f'number of speakers')
            ax.set_xlabel('number of samples')
            fig = ax.figure
            plt.tight_layout()
            plt.savefig(f'{fig_dir}{filename}.{self.format}')
            plt.close(fig)
            fig.clear()

        self.plot_distributions(df_speakers, type='speakers')

    def plot_distributions(self, df, type='samples'):
        fig_dir = self.util.get_path('fig_dir')+'../' # one up because of the runs 
        attributes = ast.literal_eval(self.util.config_val('EXPL', 'value_counts', False))
        dist_type = self.util.config_val('EXPL', 'dist_type', 'hist')
        for att in attributes:
            if len(att) == 1:
                self.util.debug(f'plotting {att[0]}')
                filename = f'{self.target}-{att[0]}'
                if self.util.is_categorical(df[att[0]]):
                    ax = df.groupby('class_label')[att[0]].value_counts().unstack().plot(kind='bar', stacked=True, title=f'{type} {df.shape[0]}', rot=0)
                    ax.set_ylabel(f'number of {type}')
                    ax.set_xlabel(self.target)
                else:
                    if dist_type == 'hist':
                        ax = sns.histplot(df, x=att[0], hue='class_label', kde=True)
                        ax.set_title(f'{type} {df.shape[0]}')
                        ax.set_xlabel(f'value of {att[0]}')
                        ax.set_ylabel(f'number of {type}')
                    else:
                        ax = sns.displot(df, x=att[0], hue='class_label', kind="kde", fill=True)
                        ax.fig.suptitle(f'{type} {df.shape[0]}') 
                fig = ax.figure
                plt.tight_layout()
                plt.savefig(f'{fig_dir}{filename}_{type}.{self.format}')
                plt.close(fig)
                fig.clear()
            elif len(att) == 2:
                self.util.debug(f'plotting {att}')
                att1 = att[0]
                att2 = att[1]
                filename = f'{att[0]}-{att[1]}'
                filename = f'{self.target}-{filename}'
                if self.util.is_categorical(df[att1]):
                    ax = sns.scatterplot(data=df, x=self.target, y=att2, hue=att1)                                
                elif self.util.is_categorical(df[att2]):
                    ax = sns.scatterplot(data=df, x=self.target, y=att1, hue=att2)                
                else:
                    ax = sns.scatterplot(data=df, x=att1, y=att2, hue='class_label')
                fig = ax.figure
                ax.set_title(f'{type} {df.shape[0]}')
                plt.tight_layout()
                plt.savefig(f'{fig_dir}{filename}_{type}.{self.format}')   
                plt.close(fig)
                fig.clear()
            else:
                self.util.error(f'plot value counts: target {att} has more than 2 values')


    def describe_df(self, name, df, target, filename):
        """Make a stacked barplot of samples and speakers per sex and target values. speaker, gender and target columns must be present"""
        fig_dir = self.util.get_path('fig_dir')+'../' # one up because of the runs 
        sampl_num = df.shape[0]
        sex_col = 'gender'
        if target == 'gender':
            sex_col = 'class_label'
        if self.util.exp_is_classification() and target != 'gender':
            target = 'class_label'
        if df.is_labeled:
            if df.got_gender and df.got_speaker:
                spkr_num = df.speaker.nunique()
                female_smpl_num = df[df[sex_col]=='female'].shape[0]
                male_smpl_num = df[df[sex_col]=='male'].shape[0]
                self.util.debug(f'plotting {name}: # samples: {sampl_num} (f: {female_smpl_num}, m: '+\
                    f'{male_smpl_num}), # speakers: {spkr_num}')
                # fig, axes = plt.subplots(nrows=1, ncols=2)
                fig, axes = plt.subplots(nrows=1, ncols=1)
                # df.groupby(target)['gender'].value_counts().unstack().plot(kind='bar', stacked=True, ax=axes[0], \
                #     title=f'samples ({sampl_num})')
                df.groupby(target)['gender'].value_counts().unstack().plot(kind='bar', stacked=True, \
                    title=f'samples ({sampl_num})')
                # df.groupby(target)['speaker'].nunique().plot(kind='bar', ax=axes[1], title=f'speakers ({spkr_num})')
            else:
                self.util.debug(f'plotting {name}: # samples: {sampl_num}')
                fig, axes = plt.subplots(nrows=1, ncols=1)
                df[target].value_counts().plot(kind='bar', ax=axes, \
                    title=f'samples ({sampl_num})')
            plt.tight_layout()
            plt.savefig(f'{fig_dir}{filename}.{self.format}')
            fig.clear()
            plt.close(fig)

    def scatter_plot(self, feats, labels, dimred_type):
        fig_dir = self.util.get_path('fig_dir')+'../' # one up because of the runs 
        sample_selection = self.util.config_val('EXPL', 'sample_selection', 'all')
        filename = self.util.get_exp_name()+sample_selection+'_'+dimred_type
        filename = f'{fig_dir}{filename}.{self.format}'
        self.util.debug(f'computing {dimred_type}, this might take a while...')
        data = None
        if dimred_type == 'tsne':
            data = self.getTsne(feats)
        elif dimred_type == 'umap':
            import umap
            y_umap = umap.UMAP(n_neighbors=10, random_state=0,).fit_transform(feats.values)
            data = pd.DataFrame(
                y_umap,
                feats.index,
                columns=['Dim_1', 'Dim_2'],
            )
        elif dimred_type == 'pca':
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            pca = PCA(n_components=2)
            y_pca = pca.fit_transform(scaler.fit_transform(feats.values))
            data = pd.DataFrame(
                y_pca,
                feats.index,
                columns=['Dim_1', 'Dim_2'],
            )
        else:
            self.util.error(f'no such dimensionaity reduction functin: {dimred_type}')
        plot_data = np.vstack((data.T, labels)).T
        plot_df = pd.DataFrame(data=plot_data, columns=('Dim_1', 'Dim_2', 'label'))
        plt.tight_layout()
        ax = sns.FacetGrid(plot_df, hue='label', height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
        fig = ax.figure
        plt.savefig(filename)
        fig.clear()
        plt.close(fig)

    def plotTsne(self, feats, labels, filename, perplexity=30, learning_rate=200):
        """Make a TSNE plot to see whether features are useful for classification"""
        fig_dir = self.util.get_path('fig_dir')+'../' # one up because of the runs 
        filename = f'{fig_dir}{filename}.{self.format}'
        self.util.debug(f'plotting tsne to {filename}, this might take a while...')
        model = TSNE(n_components=2, random_state=0, perplexity=perplexity, learning_rate=learning_rate)
        tsne_data = model.fit_transform(feats)
        tsne_data_labs = np.vstack((tsne_data.T, labels)).T
        tsne_df = pd.DataFrame(data=tsne_data_labs, columns=('Dim_1', 'Dim_2', 'label'))
        plt.tight_layout()
        ax = sns.FacetGrid(tsne_df, hue='label', height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
        fig = ax.figure
        plt.savefig(filename)
        fig.clear()
        plt.close(fig)

    def getTsne(self, feats, perplexity=30, learning_rate=200):
        """Make a TSNE plot to see whether features are useful for classification"""
        model = TSNE(n_components=2, random_state=0, perplexity=perplexity, learning_rate=learning_rate)
        tsne_data = model.fit_transform(feats)
        return tsne_data

    def plot_feature(self, title, feature, label, df_labels, df_features):
        fig_dir = self.util.get_path('fig_dir')+'../' # one up because of the runs 
        filename = f'{fig_dir}feat_dist_{title}_{feature}.{self.format}'
        df_plot =  pd.DataFrame({label:df_labels[label], feature:df_features[feature]})
        ax = sns.violinplot(data=df_plot, x=label, y=feature)
        label = self.util.config_val('DATA', 'target', 'class_label')
        ax.set(title=f'{title} samples', xlabel = label)
        fig = ax.figure
        plt.tight_layout()
        plt.savefig(filename)
        fig.clear()
        plt.close(fig)

    def plot_tree(self, model, features):
        from sklearn import tree
        ax = plt.gca()
        ax.figure.set_size_inches(100, 60)
#        tree.plot_tree(model, ax = ax)
        tree.plot_tree(model, feature_names=features.columns, ax = ax)
        plt.tight_layout()
        #print(ax)
        fig_dir = self.util.get_path('fig_dir')+'../' # one up because of the runs 
        exp_name = self.util.get_exp_name(only_data=True)
        format = self.util.config_val('PLOT', 'format', 'png')
        filename = f'{fig_dir}{exp_name}EXPL_tree-plot.{format}'
        fig = ax.figure
        fig.savefig(filename)
        fig.clear()
        plt.close(fig)