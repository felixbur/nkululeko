# plots.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import ast
from scipy import stats
from nkululeko.utils.util import Util
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
            # plt.tight_layout()
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
            # plt.tight_layout()
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
        for att in attributes:
            if len(att) == 1:
                att1 = att[0]
                if att1 == self.target:
                    self.util.debug(f"no need to correlate {att1} with itself")
                    return
                if att1 not in df:
                    self.util.error(f"unknown feature: {att1}")
                att1, df = self._check_binning(att1, df)
                class_label, df = self._check_binning("class_label", df)
                self.util.debug(f"plotting {att1}")
                filename = f"{self.target}-{att1}"
                if self.util.is_categorical(df[class_label]):
                    if self.util.is_categorical(df[att1]):
                        ax, caption = self._plot2cat(
                            df, class_label, att1, self.target, type_s
                        )
                    else:
                        ax, caption = self._plotcatcont(
                            df, class_label, att1, att1, type_s
                        )
                else:
                    if self.util.is_categorical(df[att1]):
                        ax, caption = self._plotcatcont(
                            df, att1, class_label, att1, type_s
                        )
                    else:
                        ax, caption = self._plot2cont(df, class_label, att1, type_s)
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
                att1 = att[0]
                att2 = att[1]
                if att1 == self.target or att2 == self.target:
                    self.util.debug(f"no need to correlate {self.target} with itself")
                    return
                if att1 not in df:
                    self.util.error(f"unknown feature: {att1}")
                if att2 not in df:
                    self.util.error(f"unknown feature: {att2}")
                att1, df = self._check_binning(att1, df)
                att2, df = self._check_binning(att2, df)
                self.util.debug(f"plotting {att}")
                filename = f"{att1}-{att2}"
                filename = f"{self.target}-{filename}"
                if self.util.is_categorical(df["class_label"]):
                    if self.util.is_categorical(df[att1]):
                        if self.util.is_categorical(df[att2]):
                            # class_label = cat, att1 = cat, att2 = cat
                            ax, caption = self._plot2cat(df, att1, att2, att1, type_s)
                        else:
                            # class_label = cat, att1 = cat, att2 = cont
                            ax, caption = self._plotcatcont(
                                df, att1, att2, att1, type_s
                            )
                    else:
                        if self.util.is_categorical(df[att2]):
                            # class_label = cat, att1 = cont, att2 = cat
                            ax, caption = self._plotcatcont(
                                df, att2, att1, att2, type_s
                            )
                        else:
                            # class_label = cat, att1 = cont, att2 = cont
                            ax, caption = self._plot2cont_cat(
                                df, att1, att2, "class_label", type_s
                            )
                else:  # class_label is continuous
                    if self.util.is_categorical(df[att1]):
                        if self.util.is_categorical(df[att2]):
                            # class_label = cont, att1 = cat, att2 = cat
                            ax, caption = self._plot2cat(df, att1, att2, att1, type_s)
                        else:
                            # class_label = cont, att1 = cat, att2 = cont
                            ax, caption = self._plot2cont_cat(
                                df, att2, "class_label", att1, type_s
                            )
                    else:
                        if self.util.is_categorical(df[att2]):
                            # class_label = cont, att1 = cont, att2 = cat
                            ax, caption = self._plot2cont_cat(
                                df, att1, "class_label", att2, type_s
                            )
                        else:
                            # class_label = cont, att1 = cont, att2 = cont
                            ax, caption = self._plot2cont(df, att1, att2, type_s)

                fig = ax.figure
                # avoid warning
                # plt.tight_layout()
                img_path = f"{fig_dir}{filename}_{type_s}.{self.format}"
                plt.savefig(img_path)
                plt.close(fig)
                # fig.clear()   # avoid error
                glob_conf.report.add_item(
                    ReportItem(
                        Header.HEADER_EXPLORE,
                        f"Correlation of {att1} and {att2}",
                        caption,
                        img_path,
                    )
                )
            else:
                self.util.error(
                    "plot value counts: the plot distribution descriptor for"
                    f" {att} has more than 2 values"
                )

    def _check_binning(self, att, df):
        bin_reals_att = eval(self.util.config_val("EXPL", f"{att}.bin_reals", "False"))
        if bin_reals_att:
            self.util.debug(f"binning continuous variable {att} to categories")
            att_new = f"{att}_binned"
            df[att_new] = self.util.continuous_to_categorical(df[att]).values
            att = att_new
        return att, df

    def _plot2cont_cat(self, df, cont1, cont2, cat, ylab):
        """
        plot relation of two continuous distributions with one categorical
        """
        pearson = stats.pearsonr(df[cont1], df[cont2])
        # trunc to three digits
        pearson = int(pearson[0] * 1000) / 1000
        pearson_string = f"PCC: {pearson}"
        ax = sns.lmplot(data=df, x=cont1, y=cont2, hue=cat)
        caption = f"{ylab} {df.shape[0]}. {pearson_string}"
        ax.fig.suptitle(caption)
        return ax, caption

    def _plot2cont(self, df, col1, col2, ylab):
        """
        plot relation of two continuous distributions
        """
        pearson = stats.pearsonr(df[col1], df[col2])
        # trunc to three digits
        pearson = int(pearson[0] * 1000) / 1000
        pearson_string = f"PCC: {pearson}"
        ax = sns.lmplot(data=df, x=col1, y=col2)
        caption = f"{ylab} {df.shape[0]}. {pearson_string}"
        ax.fig.suptitle(caption)
        return ax, caption

    def _plotcatcont(self, df, cat_col, cont_col, xlab, ylab):
        """
        plot relation of categorical distribution with continuous
        """
        dist_type = self.util.config_val("EXPL", "dist_type", "kde")
        cats, cat_str, es = su.get_effect_size(df, cat_col, cont_col)
        if dist_type == "hist":
            ax = sns.histplot(df, x=cont_col, hue=cat_col, kde=True)
            caption = f"{ylab} {df.shape[0]}. {cat_str} ({cats}):" f" {es}"
            ax.set_title(caption)
            ax.set_xlabel(f"{cont_col}")
            ax.set_ylabel(f"number of {ylab}")
        else:
            ax = sns.displot(
                df, x=cont_col, hue=cat_col, kind="kde", fill=True, warn_singular=False
            )
            ax.set(xlabel=f"{cont_col}")
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
            df.groupby(col1, observed=False)[col2]
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
        # plt.tight_layout()
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
            # plt.tight_layout()
            img_path = f"{fig_dir}{filename}.{self.format}"
            plt.savefig(img_path)
            fig.clear()
            plt.close(fig)
            glob_conf.report.add_item(
                ReportItem(
                    Header.HEADER_EXPLORE,
                    f"Overview on {df.shape[0]} samples",
                    "",
                    img_path,
                )
            )

    def scatter_plot(self, feats, label_df, label, dimred_type):
        fig_dir = self.util.get_path("fig_dir") + "../"  # one up because of the runs
        sample_selection = self.util.config_val("EXPL", "sample_selection", "all")
        filename = (
            f"{label}_{self.util.get_feattype_name()}_{sample_selection}_{dimred_type}"
        )
        filename = f"{fig_dir}{filename}.{self.format}"
        self.util.debug(f"computing {dimred_type}, this might take a while...")
        data = None
        labels = label_df[label]
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
            self.util.error(f"no such dimensionality reduction function: {dimred_type}")
        plot_data = np.vstack((data.T, labels)).T
        plot_df = pd.DataFrame(data=plot_data, columns=("Dim_1", "Dim_2", "label"))
        # plt.tight_layout()
        ax = (
            sns.FacetGrid(plot_df, hue="label", height=6)
            .map(plt.scatter, "Dim_1", "Dim_2")
            .add_legend()
        )
        fig = ax.figure
        plt.savefig(filename)
        fig.clear()
        plt.close(fig)
        glob_conf.report.add_item(
            ReportItem(
                Header.HEADER_EXPLORE,
                f"Scatter plot",
                f"using {dimred_type}",
                filename,
            )
        )

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
        # plt.tight_layout()
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
        # remove fullstops in the name
        feature_name = feature.replace(".", "-")
        fig_dir = self.util.get_path("fig_dir") + "../"  # one up because of the runs
        filename = f"{fig_dir}feat_dist_{title}_{feature_name}.{self.format}"
        if self.util.is_categorical(df_labels[label]):
            df_plot = pd.DataFrame(
                {label: df_labels[label], feature: df_features[feature]}
            )
            ax = sns.violinplot(data=df_plot, x=label, y=feature)
            label = self.util.config_val("DATA", "target", "class_label")
            ax.set(title=f"{title} samples", xlabel=label)
        else:
            plot_df = pd.concat([df_labels, df_features], axis=1)
            ax, caption = self._plot2cont(plot_df, label, feature, feature)
        # def _plot2cont(self, df, col1, col2, xlab, ylab):

        fig = ax.figure
        # plt.tight_layout()
        plt.savefig(filename)
        fig.clear()
        plt.close(fig)
        caption = f"Feature plot for feature {feature}"
        content = caption
        glob_conf.report.add_item(
            ReportItem(
                Header.HEADER_EXPLORE,
                caption,
                content,
                filename,
            )
        )

    def plot_tree(self, model, features):
        from sklearn import tree

        ax = plt.gca()
        ax.figure.set_size_inches(100, 60)
        #        tree.plot_tree(model, ax = ax)
        tree.plot_tree(model, feature_names=list(features.columns), ax=ax)
        # plt.tight_layout()
        # print(ax)
        fig_dir = self.util.get_path("fig_dir") + "../"  # one up because of the runs
        exp_name = self.util.get_exp_name(only_data=True)
        format = self.util.config_val("PLOT", "format", "png")
        filename = f"{fig_dir}{exp_name}EXPL_tree-plot.{format}"
        fig = ax.figure
        fig.savefig(filename)
        fig.clear()
        plt.close(fig)
        glob_conf.report.add_item(
            ReportItem(
                Header.HEADER_EXPLORE,
                f"Tree plot",
                f"for feature importance",
                filename,
            )
        )
