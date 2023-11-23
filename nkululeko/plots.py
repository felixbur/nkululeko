# plots.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import ast
from scipy import stats
from nkululeko.util import Util
import nkululeko.utils.stats as su
import nkululeko.glob_conf as glob_conf
from nkululeko.reporting.report_item import ReportItem
from nkululeko.reporting.defines import Header


class Plots:
    def __init__(self):
        """Initializing the util system"""
        self.util = Util("plots")
        self.format = self.util.config_val("PLOT", "format", "png")
        self.target = self.util.config_val("DATA", "target", "emotion")

    def plot_distributions_speaker(self, df):
        df_speakers = pd.DataFrame()
        pd.options.mode.chained_assignment = None  # default='warn'
        for s in df.speaker.unique():
            df_speaker = df[df.speaker == s]
            df_speaker["samplenum"] = df_speaker.shape[0]
            df_speakers = pd.concat([df_speakers, df_speaker.head(1)])
        # plot the distribution of samples per speaker
        fig_dir = self.util.get_path("fig_dir") + "../"  # one up because of the runs
        self.util.debug(f"plotting samples per speaker")
        if "gender" in df_speakers:
            filename = f"samples_value_counts"
            ax = (
                df_speakers.groupby("samplenum")["gender"]
                .value_counts()
                .unstack()
                .plot(
                    kind="bar",
                    stacked=True,
                    title=f"samples per speaker ({df_speakers.shape[0]})",
                    rot=0,
                )
            )
            ax.set_ylabel(f"number of speakers")
            ax.set_xlabel("number of samples")
            fig = ax.figure
            plt.tight_layout()
            img_path = f"{fig_dir}{filename}.{self.format}"
            plt.savefig(img_path)
            plt.close(fig)
            glob_conf.report.add_item(
                ReportItem(
                    Header.HEADER_EXPLORE,
                    "Samples per speaker",
                    f"Samples per speaker ({df_speakers.shape[0]})",
                    img_path,
                )
            )
            # fig.clear()
        else:
            filename = f"samples_value_counts"
            ax = (
                df_speakers["samplenum"]
                .value_counts()
                .sort_values()
                .plot(
                    kind="bar",
                    stacked=True,
                    title=f"samples per speaker ({df_speakers.shape[0]})",
                    rot=0,
                )
            )
            ax.set_ylabel(f"number of speakers")
            ax.set_xlabel("number of samples")
            fig = ax.figure
            plt.tight_layout()
            img_path = f"{fig_dir}{filename}.{self.format}"
            plt.savefig(img_path)
            plt.close(fig)
            fig.clear()
            glob_conf.report.add_item(
                ReportItem(
                    Header.HEADER_EXPLORE,
                    "Sample value counts",
                    f"Samples per speaker ({df_speakers.shape[0]})",
                    img_path,
                )
            )
        self.plot_distributions(df_speakers, type_s="speakers")

    def plot_distributions(self, df, type_s="samples"):
        fig_dir = self.util.get_path("fig_dir") + "../"  # one up because of the runs
        attributes = ast.literal_eval(
            self.util.config_val("EXPL", "value_counts", False)
        )
        bin_reals = eval(self.util.config_val("EXPL", "bin_reals", "False"))
        for att in attributes:
            if len(att) == 1:
                if att[0] not in df:
                    self.util.error(f"unknown feature: {att[0]}")
                self.util.debug(f"plotting {att[0]}")
                filename = f"{self.target}-{att[0]}"
                if self.util.is_categorical(df[att[0]]):
                    if self.util.is_categorical(df["class_label"]):
                        ax, caption = self._plot2cat(
                            df, "class_label", att[0], self.target, type_s
                        )
                    else:
                        ax, caption = self._plotcatcont(
                            df, "class_label", att[0], self.target, type_s
                        )
                else:
                    if self.util.is_categorical(df["class_label"]) or bin_reals:
                        if bin_reals:
                            self.util.debug(
                                f"{self.name}: binning continuous variable to categories"
                            )
                            cat_vals = self.util.continuous_to_categorical(
                                df[self.target]
                            )
                            df[f"{self.target}_binned"] = cat_vals
                            ax, caption = self._plotcatcont(
                                df,
                                f"{self.target}_binned",
                                att[0],
                                self.target,
                                type_s,
                            )
                        else:
                            ax, caption = self._plotcatcont(
                                df, att[0], "class_label", att[0], type_s
                            )
                    else:
                        ax, caption = self._plot2cont(
                            df, self.target, att[0], self.target, type_s
                        )
                fig = ax.figure
                # plt.tight_layout()
                img_path = f"{fig_dir}{filename}_{type_s}.{self.format}"
                plt.savefig(img_path)
                plt.close(fig)
                glob_conf.report.add_item(
                    ReportItem(
                        Header.HEADER_EXPLORE,
                        f"Correlation of {self.target} and {att[0]}",
                        caption,
                        img_path,
                    )
                )

                # fig.clear()           # avoid error
            elif len(att) == 2:
                if att[0] not in df:
                    self.util.error(f"unknown feature: {att[0]}")
                if att[1] not in df:
                    self.util.error(f"unknown feature: {att[1]}")
                self.util.debug(f"plotting {att}")
                att1 = att[0]
                att2 = att[1]
                filename = f"{att1}-{att2}"
                filename = f"{self.target}-{filename}"
                pearson_string = ""
                if self.util.is_categorical(df[att1]):
                    ax = sns.scatterplot(data=df, x=self.target, y=att2, hue=att1)
                elif self.util.is_categorical(df[att2]):
                    ax = sns.scatterplot(data=df, x=self.target, y=att1, hue=att2)
                else:
                    pearson = stats.pearsonr(df[att1], df[att2])
                    pearson = int(pearson[0] * 1000) / 1000
                    pearson_string = f"PCC: {pearson}"
                    ax = sns.scatterplot(data=df, x=att1, y=att2, hue="class_label")
                fig = ax.figure
                ax.set_title(f"{type_s} {df.shape[0]}. {pearson_string}")
                plt.tight_layout()
                plt.savefig(f"{fig_dir}{filename}_{type}.{self.format}")
                plt.close(fig)
                # fig.clear()   # avoid error
            else:
                self.util.error(
                    "plot value counts: the plot distribution descriptor for"
                    f" {att} has more than 2 values"
                )

    def _plot2cont(self, df, col1, col2, xlab, ylab):
        """
        plot relation of two continuous distributions
        """
        pearson = stats.pearsonr(df[col1], df[col2])
        # trunc to three digits
        pearson = int(pearson[0] * 1000) / 1000
        pearson_string = f"PCC: {pearson}"
        ax = sns.scatterplot(data=df, x=col1, y=col2)
        caption = f"{ylab} {df.shape[0]}. {pearson_string}"
        ax.set_title(caption)
        return ax, caption

    def _plotcatcont(self, df, cat_col, cont_col, xlab, ylab):
        """
        plot relation of categorical distribution with continuous
        """
        dist_type = self.util.config_val("EXPL", "dist_type", "kde")
        cats, cat_str, es = su.get_effect_size(df, cont_col, cat_col)
        if dist_type == "hist":
            ax = sns.histplot(df, x=cat_col, hue=cont_col, kde=True)
            caption = f"{ylab} {df.shape[0]}. {cat_str} ({cats}):" f" {es}"
            ax.set_title(caption)
            ax.set_xlabel(f"{xlab}")
            ax.set_ylabel(f"number of {ylab}")
        else:
            ax = sns.displot(df, x=cat_col, hue=cont_col, kind="kde", fill=True)
            ax.set(xlabel=f"{xlab}")
            caption = f"{ylab} {df.shape[0]}. {cat_str} ({cats}):" f" {es}"
            ax.fig.suptitle(caption)
        return ax, caption

    def _plot2cat(self, df, col1, col2, xlab, ylab):
        """
        plot relation of 2 categorical distributions
        """
        crosstab = pd.crosstab(index=df[col1], columns=df[col2])
        res_pval = stats.chi2_contingency(crosstab)
        res_pval = int(res_pval[1] * 1000) / 1000
        caption = f"{ylab} {df.shape[0]}. P-val chi2: {res_pval}"
        ax = (
            df.groupby(col1)[col2]
            .value_counts()
            .unstack()
            .plot(kind="bar", stacked=True, title=caption, rot=0)
        )
        ax.set_ylabel(f"number of {ylab}")
        ax.set_xlabel(xlab)
        return ax, caption

    def plot_durations(self, df, filename, sample_selection, caption=""):
        fig_dir = self.util.get_path("fig_dir") + "../"  # one up because of the runs
        try:
            ax = sns.histplot(df, x="duration", hue="class_label", kde=True)
        except AttributeError as ae:
            self.util.warn(ae)
            ax = sns.histplot(df, x="duration", kde=True)
        title = f"Duration distribution for {sample_selection} {df.shape[0]}"
        ax.set_title(title)
        ax.set_xlabel(f"duration")
        ax.set_ylabel(f"number of samples")
        fig = ax.figure
        plt.tight_layout()
        img_path = f"{fig_dir}{filename}_{sample_selection}.{self.format}"
        plt.savefig(img_path)
        plt.close(fig)
        glob_conf.report.add_item(
            ReportItem(
                Header.HEADER_EXPLORE,
                caption,
                title,
                img_path,
            )
        )

    def describe_df(self, name, df, target, filename):
        """Make a stacked barplot of samples and speakers per sex and target values. speaker, gender and target columns must be present"""
        fig_dir = self.util.get_path("fig_dir") + "../"  # one up because of the runs
        sampl_num = df.shape[0]
        sex_col = "gender"
        if target == "gender":
            sex_col = "class_label"
        if self.util.exp_is_classification() and target != "gender":
            target = "class_label"
        if df.is_labeled:
            if df.got_gender and df.got_speaker:
                spkr_num = df.speaker.nunique()
                female_smpl_num = df[df[sex_col] == "female"].shape[0]
                male_smpl_num = df[df[sex_col] == "male"].shape[0]
                self.util.debug(
                    f"plotting {name}: # samples: {sampl_num} (f:"
                    f" {female_smpl_num}, m: "
                    + f"{male_smpl_num}), # speakers: {spkr_num}"
                )
                # fig, axes = plt.subplots(nrows=1, ncols=2)
                fig, axes = plt.subplots(nrows=1, ncols=1)
                # df.groupby(target)['gender'].value_counts().unstack().plot(kind='bar', stacked=True, ax=axes[0], \
                #     title=f'samples ({sampl_num})')
                df.groupby(target)["gender"].value_counts().unstack().plot(
                    kind="bar", stacked=True, title=f"samples ({sampl_num})"
                )
                # df.groupby(target)['speaker'].nunique().plot(kind='bar', ax=axes[1], title=f'speakers ({spkr_num})')
            else:
                self.util.debug(f"plotting {name}: # samples: {sampl_num}")
                fig, axes = plt.subplots(nrows=1, ncols=1)
                df[target].value_counts().plot(
                    kind="bar", ax=axes, title=f"samples ({sampl_num})"
                )
            plt.tight_layout()
            plt.savefig(f"{fig_dir}{filename}.{self.format}")
            fig.clear()
            plt.close(fig)

    def scatter_plot(self, feats, labels, dimred_type):
        fig_dir = self.util.get_path("fig_dir") + "../"  # one up because of the runs
        sample_selection = self.util.config_val("EXPL", "sample_selection", "all")
        filename = self.util.get_exp_name() + sample_selection + "_" + dimred_type
        filename = f"{fig_dir}{filename}.{self.format}"
        self.util.debug(f"computing {dimred_type}, this might take a while...")
        data = None
        if dimred_type == "tsne":
            data = self.getTsne(feats)
        elif dimred_type == "umap":
            import umap

            y_umap = umap.UMAP(
                n_neighbors=10,
                random_state=0,
            ).fit_transform(feats.values)
            data = pd.DataFrame(
                y_umap,
                feats.index,
                columns=["Dim_1", "Dim_2"],
            )
        elif dimred_type == "pca":
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            pca = PCA(n_components=2)
            y_pca = pca.fit_transform(scaler.fit_transform(feats.values))
            data = pd.DataFrame(
                y_pca,
                feats.index,
                columns=["Dim_1", "Dim_2"],
            )
        else:
            self.util.error(f"no such dimensionaity reduction functin: {dimred_type}")
        plot_data = np.vstack((data.T, labels)).T
        plot_df = pd.DataFrame(data=plot_data, columns=("Dim_1", "Dim_2", "label"))
        plt.tight_layout()
        ax = (
            sns.FacetGrid(plot_df, hue="label", height=6)
            .map(plt.scatter, "Dim_1", "Dim_2")
            .add_legend()
        )
        fig = ax.figure
        plt.savefig(filename)
        fig.clear()
        plt.close(fig)

    def plotTsne(self, feats, labels, filename, perplexity=30, learning_rate=200):
        """Make a TSNE plot to see whether features are useful for classification"""
        fig_dir = self.util.get_path("fig_dir") + "../"  # one up because of the runs
        filename = f"{fig_dir}{filename}.{self.format}"
        self.util.debug(f"plotting tsne to {filename}, this might take a while...")
        model = TSNE(
            n_components=2,
            random_state=0,
            perplexity=perplexity,
            learning_rate=learning_rate,
        )
        tsne_data = model.fit_transform(feats)
        tsne_data_labs = np.vstack((tsne_data.T, labels)).T
        tsne_df = pd.DataFrame(data=tsne_data_labs, columns=("Dim_1", "Dim_2", "label"))
        plt.tight_layout()
        ax = (
            sns.FacetGrid(tsne_df, hue="label", height=6)
            .map(plt.scatter, "Dim_1", "Dim_2")
            .add_legend()
        )
        fig = ax.figure
        plt.savefig(filename)
        fig.clear()
        plt.close(fig)

    def getTsne(self, feats, perplexity=30, learning_rate=200):
        """Make a TSNE plot to see whether features are useful for classification"""
        model = TSNE(
            n_components=2,
            random_state=0,
            perplexity=perplexity,
            learning_rate=learning_rate,
        )
        tsne_data = model.fit_transform(feats)
        return tsne_data

    def plot_feature(self, title, feature, label, df_labels, df_features):
        fig_dir = self.util.get_path("fig_dir") + "../"  # one up because of the runs
        filename = f"{fig_dir}feat_dist_{title}_{feature}.{self.format}"
        if self.util.is_categorical(df_labels[label]):
            df_plot = pd.DataFrame(
                {label: df_labels[label], feature: df_features[feature]}
            )
            ax = sns.violinplot(data=df_plot, x=label, y=feature)
            label = self.util.config_val("DATA", "target", "class_label")
            ax.set(title=f"{title} samples", xlabel=label)
        else:
            plot_df = pd.concat([df_labels, df_features], axis=1)
            ax, caption = self._plot2cont(plot_df, label, feature, label, feature)
        # def _plot2cont(self, df, col1, col2, xlab, ylab):

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
        tree.plot_tree(model, feature_names=list(features.columns), ax=ax)
        plt.tight_layout()
        # print(ax)
        fig_dir = self.util.get_path("fig_dir") + "../"  # one up because of the runs
        exp_name = self.util.get_exp_name(only_data=True)
        format = self.util.config_val("PLOT", "format", "png")
        filename = f"{fig_dir}{exp_name}EXPL_tree-plot.{format}"
        fig = ax.figure
        fig.savefig(filename)
        fig.clear()
        plt.close(fig)
