# plots.py
import ast
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.manifold import TSNE

import audeer
from audmetric import concordance_cc as ccc

import nkululeko.glob_conf as glob_conf
from nkululeko.reporting.defines import Header
from nkululeko.reporting.report_item import ReportItem
import nkululeko.utils.stats as su
from nkululeko.utils.util import Util


class Plots:
    def __init__(self):
        """Initializing the util system."""
        self.util = Util("plots")
        self.format = self.util.config_val("PLOT", "format", "png")
        self.target = self.util.config_val("DATA", "target", "emotion")
        self.with_ccc = eval(self.util.config_val("PLOT", "ccc", "False"))
        self.type_s = "samples"
        self.titles = eval(self.util.config_val("PLOT", "titles", "True"))
        self.print_stats = eval(self.util.config_val("EXPL", "print_stats", "False"))

    def plot_distributions_speaker(self, df: pd.DataFrame):
        title = ""
        if df.empty:
            self.util.warn(
                "plot_distributions_speaker: empty DataFrame, nothing to plot"
            )
            return
        self.type_s = "speaker"
        df_speakers = pd.DataFrame()
        pd.options.mode.chained_assignment = None  # default='warn'
        for s in df.speaker.unique():
            df_speaker = df[df.speaker == s]
            df_speaker["samplenum"] = df_speaker.shape[0]
            df_speakers = pd.concat([df_speakers, df_speaker.head(1)])
        # plot the distribution of samples per speaker
        self.util.debug("plotting samples per speaker")
        if "gender" in df_speakers:
            filename = "samples_value_counts"
            if self.titles:
                title = f"samples per speaker ({df_speakers.shape[0]})"
            ax = (
                df_speakers.groupby("samplenum")["gender"]
                .value_counts()
                .unstack()
                .plot(
                    kind="bar",
                    stacked=True,
                    title=title,
                    rot=0,
                )
            )
            ax.set_ylabel("number of speakers")
            ax.set_xlabel("number of samples")
            self.save_plot(
                ax,
                "Samples per speaker",
                f"Samples per speaker ({df_speakers.shape[0]})",
                filename,
                "speakers",
            )

            # fig.clear()
        else:
            filename = "samples_value_counts"
            if self.titles:
                title = f"samples per speaker ({df_speakers.shape[0]})"
            ax = (
                df_speakers["samplenum"]
                .value_counts()
                .sort_values()
                .plot(
                    kind="bar",
                    stacked=True,
                    title=title,
                    rot=0,
                )
            )
            ax.set_ylabel("number of speakers")
            ax.set_xlabel("number of samples")
            self.save_plot(
                ax,
                "Sample value counts",
                f"Samples per speaker ({df_speakers.shape[0]})",
                filename,
                "speakers",
            )

        self.plot_distributions(df_speakers, type_s="speakers")

    def plot_distributions(self, df: pd.DataFrame, type_s: str = "samples"):
        if df.empty:
            self.util.warn("plot_distributions: empty DataFrame, nothing to plot")
            return
        class_label, df = self._check_binning("class_label", df)
        value_counts_conf = self.util.config_val("EXPL", "value_counts", False)
        if not isinstance(value_counts_conf, str):
            value_counts_conf = str(value_counts_conf)
        attributes = ast.literal_eval(value_counts_conf)
        # always plot the distribution of the main attribute
        filename = f"{class_label}_distribution"
        if self.util.is_categorical(df[class_label]):
            ax = df[class_label].value_counts().plot(kind="bar")
        else:
            # for continuous variables, also add a discretized version
            binned_data = self.util.continuous_to_categorical(df[class_label])
            ax = binned_data.value_counts().plot(kind="bar")
            filename_binned = f"{class_label}_discreet"
            self.save_plot(
                ax,
                "Sample value counts",
                filename_binned,
                filename_binned,
                type_s,
            )
            dist_type = self.util.config_val("EXPL", "dist_type", "hist")
            ax = df[class_label].plot(kind=dist_type)

        self.save_plot(
            ax,
            "Sample value counts",
            filename,
            filename,
            type_s,
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
                self.util.debug(f"plotting {att1}")
                filename = f"{self.target}-{att1}"
                if self.util.is_categorical(df[class_label]):
                    if self.util.is_categorical(df[att1]):
                        ax, caption = self._plot2cat(
                            df, class_label, att1, self.target, type_s
                        )
                    else:
                        ax, caption = self.plotcatcont(
                            df, class_label, att1, att1, type_s
                        )
                else:
                    if self.util.is_categorical(df[att1]):
                        ax, caption = self.plotcatcont(
                            df, att1, class_label, att1, type_s
                        )
                    else:
                        ax, caption = self._plot2cont(df, class_label, att1, type_s)
                self.save_plot(
                    ax,
                    caption,
                    f"Correlation of {self.target} and {att[0]}",
                    filename,
                    type_s,
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
                            ax, caption = self.plotcatcont(df, att1, att2, att1, type_s)
                    else:
                        if self.util.is_categorical(df[att2]):
                            # class_label = cat, att1 = cont, att2 = cat
                            ax, caption = self.plotcatcont(df, att2, att1, att2, type_s)
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

                self.save_plot(
                    ax, caption, f"Correlation of {att1} and {att2}", filename, type_s
                )

            else:
                self.util.error(
                    "plot value counts: the plot distribution descriptor for"
                    f" {att} has more than 2 values. Perhaps you forgot to state a list of lists?"
                )

    def save_plot(self, ax, caption, header, filename, type_s):
        # one up because of the runs
        fig_dir = audeer.path(self.util.get_path("fig_dir"), "..")
        fig_plots = ax.figure
        # avoid warning
        # plt.tight_layout()
        img_path = os.path.join(fig_dir, f"{filename}_{type_s}.{self.format}")
        plt.savefig(img_path)
        plt.close(fig_plots)
        self.util.debug(f"Saved plot to {img_path}")
        glob_conf.report.add_item(
            ReportItem(
                Header.HEADER_EXPLORE,
                header,
                caption,
                img_path,
            )
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
        """Plot relation of two continuous distributions with one categorical."""
        plot_df = df[[cont1, cont2, cat]].copy()
        if cont2 == "class_label":
            plot_df = plot_df.rename(columns={cont2: self.target})
            cont2 = self.target
        if cont1 == "class_label":
            plot_df = plot_df.rename(columns={cont1: self.target})
            cont1 = self.target
        if cat == "class_label":
            plot_df = plot_df.rename(columns={cat: self.target})
            cat = self.target
        pearson = stats.pearsonr(plot_df[cont1], plot_df[cont2])
        # trunc to three digits
        pearson = int(pearson[0] * 1000) / 1000
        pearson_string = f"PCC: {pearson}"
        ccc_string = ""
        if self.with_ccc:
            ccc_val = ccc(plot_df[cont1], plot_df[cont2])
            ccc_val = int(ccc_val * 1000) / 1000
            ccc_string = f"CCC: {ccc_val}"
        ax = sns.lmplot(data=plot_df, x=cont1, y=cont2, hue=cat)
        caption = f"{ylab} {plot_df.shape[0]}. {pearson_string} {ccc_string}"
        ax.figure.suptitle(caption)
        return ax, caption

    def _plot2cont(self, df, col1, col2, ylab):
        """Plot relation of two continuous distributions."""
        plot_df = df[[col1, col2]].copy()
        # rename "class_label" to the original target
        if col2 == "class_label":
            plot_df = plot_df.rename(columns={col2: self.target})
            col2 = self.target
        if col1 == "class_label":
            plot_df = plot_df.rename(columns={col1: self.target})
            col1 = self.target
        pearson = stats.pearsonr(plot_df[col1], plot_df[col2])
        # trunc to three digits
        pearson = int(pearson[0] * 1000) / 1000
        pearson_string = f"PCC: {pearson}"
        ccc_string = ""
        if self.with_ccc:
            ccc_val = ccc(plot_df[col1], plot_df[col2])
            ccc_val = int(ccc_val * 1000) / 1000
            ccc_string = f"CCC: {ccc_val}"
        ax = sns.lmplot(data=plot_df, x=col1, y=col2)
        caption = f"{ylab} {plot_df.shape[0]}. {pearson_string} {ccc_string}"
        ax.figure.suptitle(caption)
        return ax, caption

    def plotcatcont(self, df, cat_col, cont_col, xlab, ylab):
        """Plot relation of categorical distribution with continuous."""
        # rename "class_label" to the original target
        plot_df = df[[cat_col, cont_col]].copy()
        if cat_col == "class_label":
            plot_df = plot_df.rename(columns={cat_col: self.target})
            cat_col = self.target
        elif cont_col == "class_label":
            plot_df = plot_df.rename(columns={cont_col: self.target})
            cont_col = self.target
        dist_type = self.util.config_val("EXPL", "dist_type", "kde")
        fill_areas = eval(self.util.config_val("PLOT", "fill_areas", "False"))
        max_cat, cat_str, effect_results = su.get_effect_size(
            plot_df, cat_col, cont_col
        )
        self.util.debug(effect_results)
        self.util.print_results_to_store(
            f"cohens-d_{self.type_s}", str(effect_results) + "\n"
        )
        es = effect_results[max_cat]
        model_type = self.util.get_model_type()
        if dist_type == "hist" and model_type != "tree":
            ax = sns.histplot(plot_df, x=cont_col, hue=cat_col, kde=True)
            caption = f"{ylab} {plot_df.shape[0]}. {cat_str} ({max_cat}):" f" {es}"
            ax.set_title(caption)
            ax.set_xlabel(f"{cont_col}")
            ax.set_ylabel(f"number of {ylab}")
        else:
            ax = sns.displot(
                plot_df,
                x=cont_col,
                hue=cat_col,
                kind="kde",
                fill=fill_areas,
                warn_singular=False,
            )
            ax.set(xlabel=f"{cont_col}")
            caption = f"{ylab} {plot_df.shape[0]}. {cat_str} ({max_cat}):" f" {es}"
            ax.figure.suptitle(caption)
        return ax, caption

    def _plot2cat(self, df, col1, col2, xlab, ylab):
        """Plot relation of 2 categorical distributions."""
        plot_df = df[[col1, col2]].copy()
        # rename "class_label" to the original target
        if col2 == "class_label":
            plot_df = plot_df.rename(columns={col2: self.target})
            col2 = self.target
        elif col1 == "class_label":
            plot_df = plot_df.rename(columns={col1: self.target})
            col1 = self.target
        crosstab = pd.crosstab(index=plot_df[col1], columns=plot_df[col2])
        res_pval = stats.chi2_contingency(crosstab)
        res_pval = int(res_pval[1] * 1000) / 1000
        caption = f"{ylab} {plot_df.shape[0]}. P-val chi2: {res_pval}"
        ax = (
            plot_df.groupby(col1, observed=False)[col2]
            .value_counts()
            .unstack()
            .plot(kind="bar", stacked=True, title=caption, rot=0)
        )
        ax.set_ylabel(f"number of {ylab}")
        ax.set_xlabel(xlab)
        return ax, caption

    def plot_durations(self, df, filename, sample_selection, caption=""):
        title = ""
        # one up because of the runs
        fig_dir = os.path.join(self.util.get_path("fig_dir"), "..")
        try:
            ax = sns.histplot(df, x="duration", hue="class_label", kde=True)
        except AttributeError as ae:
            self.util.warn(ae)
            ax = sns.histplot(df, x="duration", kde=True)
        except ValueError as error:
            self.util.warn(error)
            ax = sns.histplot(df, x="duration", kde=True)
        min = self.util.to_3_digits(df.duration.min())
        max = self.util.to_3_digits(df.duration.max())
        if self.titles:
            title = f"Duration distr. for {sample_selection} {df.shape[0]}. min={min}, max={max}"
            ax.set_title(title)
        ax.set_xlabel("duration")
        ax.set_ylabel("number of samples")
        fig = ax.figure
        # plt.tight_layout()
        img_path = os.path.join(fig_dir, f"{filename}_{sample_selection}.{self.format}")
        plt.savefig(img_path)
        plt.close(fig)
        self.util.debug(f"plotted durations to {img_path}")
        glob_conf.report.add_item(
            ReportItem(
                Header.HEADER_EXPLORE,
                caption,
                title,
                img_path,
            )
        )

    def plot_speakers(self, df, sample_selection):
        filename = "speakers"
        caption = "speakers"
        # one up because of the runs
        fig_dir = os.path.join(self.util.get_path("fig_dir"), "..")
        sns.set_style("whitegrid")  # Set style for chart
        ax = df["speaker"].value_counts().plot(kind="pie", autopct="%1.1f%%")
        if self.titles:
            title = f"Speaker distr. for {sample_selection} {df.shape[0]}."
            ax.set_title(title)
        fig = ax.figure
        # plt.tight_layout()
        img_path = os.path.join(fig_dir, f"{filename}_{sample_selection}.{self.format}")
        plt.savefig(img_path)
        plt.close(fig)
        self.util.debug(f"plotted speakers to {img_path}")
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
        fig_dir = self.util.get_path("fig_dir")  # + "../"  # one up because of the runs
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
            img_path = os.path.join(fig_dir, f"{filename}.{self.format}")
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
        dim_num = int(self.util.config_val("EXPL", "scatter.dim", 2))
        # one up because of the runs (for explore module)
        fig_dir = os.path.join(self.util.get_path("fig_dir"), "..")
        sample_selection = self.util.config_val("EXPL", "sample_selection", "all")
        exp_name = self.util.get_name()
        filename = f"{label}_{exp_name}_{self.util.get_feattype_name()}_{sample_selection}_{dimred_type}_{str(dim_num)}d"
        filename = os.path.join(fig_dir, f"{filename}.{self.format}")
        self.util.debug(f"computing {dimred_type}, this might take a while...")
        data = None
        labels = label_df[label]
        if dimred_type == "tsne":
            data = self.getTsne(feats, dim_num)
        else:
            if dimred_type == "umap":
                import umap

                y = umap.UMAP(
                    n_neighbors=10,
                    random_state=0,
                    n_components=dim_num,
                ).fit_transform(feats.values)
            elif dimred_type == "pca":
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()
                pca = PCA(n_components=dim_num)
                y = pca.fit_transform(scaler.fit_transform(feats.values))
            else:
                self.util.error(
                    f"no such dimensionality reduction function: {dimred_type}"
                )
            if dim_num == 2:
                columns = ["Dim_1", "Dim_2"]
            elif dim_num == 3:
                columns = ["Dim_1", "Dim_2", "Dim_3"]
            else:
                self.util.error(f"wrong dimension number: {dim_num}")
            data = pd.DataFrame(
                y,
                feats.index,
                columns=columns,
            )

        if dim_num == 2:
            plot_data = np.vstack((data.T, labels)).T
            plot_df = pd.DataFrame(data=plot_data, columns=("Dim_1", "Dim_2", "label"))
            # plt.tight_layout()
            ax = (
                sns.FacetGrid(plot_df, hue="label", height=6)
                .map(plt.scatter, "Dim_1", "Dim_2")
                .add_legend()
            )
        elif dim_num == 3:
            from mpl_toolkits.mplot3d import Axes3D
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()

            labels_e = le.fit_transform(labels)
            plot_data = np.vstack((data.T, labels_e)).T
            plot_df = pd.DataFrame(
                data=plot_data, columns=("Dim_1", "Dim_2", "Dim_3", "label")
            )
            # plt.tight_layout()
            # axes instance
            fig = plt.figure(figsize=(6, 6))
            ax = Axes3D(fig, auto_add_to_figure=False)
            fig.add_axes(ax)
            # get colormap from seaborn
            # cmap = ListedColormap(sns.color_palette("hsv", 256).as_hex())
            color_dict = {
                0: "red",
                1: "blue",
                2: "green",
                3: "yellow",
                4: "purple",
                5: "#ff69b4",
                6: "black",
                7: "cyan",
                8: "magenta",
                9: "#faebd7",
                10: "#2e8b57",
                11: "#eeefff",
                12: "#da70d6",
                13: "#ff7f50",
                14: "#cd853f",
                15: "#bc8f8f",
                16: "#5f9ea0",
                17: "#daa520",
            }
            # plot
            # make the numbers bigger so they can be used as distinguishable colors
            labels_ex = [color_dict[xi] for xi in labels_e]
            sc = ax.scatter(
                plot_df.Dim_1,
                plot_df.Dim_2,
                plot_df.Dim_3,
                s=40,
                c=labels_ex,
                marker="o",
                # cmap=cmap,
                alpha=1,
            )
            ax.set_xlabel("Dim_1")
            ax.set_ylabel("Dim_2")
            ax.set_zlabel("Dim_3")
            # legend
            plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
        else:
            self.util.error(f"wrong dimension number: {dim_num}")
        fig = ax.figure
        plt.savefig(filename)
        self.util.debug(f"plotted {dimred_type} scatter plot to {filename}")
        fig.clear()
        plt.close(fig)
        glob_conf.report.add_item(
            ReportItem(
                Header.HEADER_EXPLORE,
                "Scatter plot",
                f"using {dimred_type}",
                filename,
            )
        )

    def getTsne(self, feats, dim_num, perplexity=30, learning_rate=200):
        """Make a TSNE plot to see whether features are useful for classification"""
        model = TSNE(
            n_components=dim_num,
            random_state=0,
            perplexity=perplexity,
            learning_rate=learning_rate,
        )
        tsne_data = model.fit_transform(feats)
        return tsne_data

    def plot_feature(self, title, feature, label, df_labels, df_features):
        # remove fullstops in the name
        feature_name = str(feature).replace(".", "-")
        # one up because of the runs
        fig_dir = audeer.path(self.util.get_path("fig_dir"), "..")
        filename = audeer.path(
            fig_dir, f"feat_dist_{title}_{feature_name}.{self.format}"
        )
        ignore_gender = eval(self.util.config_val("EXPL", "ignore_gender", "False"))
        sample_num = df_labels.shape[0]
        if self.util.is_categorical(df_labels[label]):
            p_val = ""
            cat_num = df_labels[label].nunique()
            if (
                "gender" in df_labels
                and df_labels["gender"].notna().any()
                and not ignore_gender
            ):
                # plot distribution for each gender in parallel violin plots
                df_plot = pd.DataFrame(
                    {
                        label: df_labels[label],
                        feature: df_features[feature],
                        "gender": df_labels["gender"],
                    }
                )
                ax = sns.violinplot(
                    data=df_plot, x=label, y=feature, hue="gender", split=True
                )
            else:
                df_plot = pd.DataFrame(
                    {label: df_labels[label], feature: df_features[feature]}
                )
                ax = sns.violinplot(data=df_plot, x=label, y=feature)
            val_dict, mean_featnum = self.util.df_to_categorical_dict(
                df_plot, label, feature
            )
            pairwise_results, overall_results = su.find_most_significant_difference(
                val_dict, mean_featnum
            )
            # 'approach', 'combo', test statistic, 'p_value', 'significance','all_results'
            if self.print_stats:
                if overall_results is not None:
                    self.util.debug(
                        f"overall results for {feature_name} from statistical test: {overall_results['all_results']}"
                    )
                if pairwise_results is not None:
                    self.util.debug(
                        f"pairwise results from statistical test: {pairwise_results['all_results']}"
                    )
            label = self.util.config_val("DATA", "target", "class_label")
            if self.titles:
                if cat_num > 2:
                    title = (
                        f"{title} samples ({sample_num})\n"
                        + f"{overall_results['approach']}: {overall_results['combo']}:"
                        f"{overall_results['significance']})\n"
                        + f"{pairwise_results['approach']}: {pairwise_results['combo']}:"
                        f"{pairwise_results['significance']})"
                    )
                else:
                    title = (
                        f"{title} samples ({sample_num})\n"
                        + f"{pairwise_results['approach']}: {pairwise_results['combo']}:"
                        f"{pairwise_results['significance']})"
                    )

                ax.set(title=title, xlabel=label)
            else:
                ax.set(xlabel=label)
        else:
            plot_df = pd.concat([df_labels, df_features], axis=1)
            ax, caption = self._plot2cont(plot_df, label, feature, feature)
        fig = ax.figure
        plt.tight_layout()
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

    def regplot(self, reglist, labels, features):
        title = "regplot"
        if len(reglist) > 1 and len(reglist) < 4:
            if len(reglist) == 2:
                cat_var = "class_label"
            else:
                cat_var = reglist[2]
            # remove fullstops in the name
            feat_x = str(reglist[0]).replace(".", "-")
            feat_y = str(reglist[1]).replace(".", "-")
            # one up because of the runs
            fig_dir = audeer.path(self.util.get_path("fig_dir"), "..")
            filename = audeer.path(
                fig_dir, f"{title}_{feat_x}-{feat_y}-{cat_var}.{self.format}"
            )
            try:
                if self.util.is_categorical(labels[cat_var]):
                    plot_df = features[[feat_x, feat_y]]
                    plot_df = pd.concat([plot_df, labels[cat_var]], axis=1)
                    # ax = sns.scatterplot(data=plot_df, x=feat_x, y=feat_y, hue=cat_var)
                    ax = sns.pairplot(data=plot_df, x_vars=feat_x, y_vars=feat_y, hue=cat_var, kind="reg", height=7)
                else: 
                    bubble_sizes = self.util.scale_to_range(labels[cat_var].values, new_min=5, new_max=50)
                    #bubble_sizes = labels[cat_var].values
                    plot_df = features[[feat_x, feat_y]]
                    plot_df = pd.concat([plot_df, labels[cat_var]], axis=1)
                    # Add bubble sizes to DataFrame for seaborn
                    plot_df['bubble_size'] = bubble_sizes
                    # Create scatter plot with seaborn
                    ax = sns.scatterplot(
                        data=plot_df,
                        x=feat_x,
                        y=feat_y,
                        size='bubble_size',
                        sizes=(5, 50),  # min and max bubble sizes
                        hue=cat_var,  # color by third variable
                        palette='viridis',
                        alpha=0.6,
                        edgecolor='black',
                        linewidth=0.5,
                    )
                    # Remove size legend, keep only color legend
                    handles, bub_labels = ax.get_legend_handles_labels()
                    # Find where size legend starts (typically after title "bubble_size")
                    if 'bubble_size' in bub_labels:
                        idx = bub_labels.index('bubble_size')
                        # Keep only handles/labels before the size legend section
                        ax.legend(handles[:idx], bub_labels[:idx])

                    plt.xlabel(feat_x, fontsize=12)
                    plt.ylabel(feat_y, fontsize=12)
                    #plt.title('Bubble Plot with Seaborn\n(Bubble size represents feature_3)', fontsize=14)
                    plt.grid(True, alpha=0.3)
            except KeyError as ke:
                r = re.compile(f".*{re.escape(ke.args[0])}.*")
                s_list = list(filter(r.match, features.columns)) 
                self.util.error(f"regplot feature not found: {ke}\nDid you mean {s_list} ?")
            pearson = stats.pearsonr(features[feat_x], features[feat_y])
            # trunc to three digits
            pearson = int(pearson[0] * 1000) / 1000
            pearson_string = f"PCC: {pearson}"

            if self.print_stats:
                import statsmodels.formula.api as smf
                # ... add "emotion" and "speaker" column to MLD feature table
                data_df = features
                data_df[self.target] = labels["class_label"].values
                data_df["speaker"] = labels["speaker"].values
                model = smf.mixedlm(f"{feat_x} ~ {self.target} * {feat_y}", 
                                    data_df, 
                                    groups=data_df["speaker"])
                result = model.fit()
                self.util.debug(result.summary())
            if self.titles:
                title = (
                    f"{title} samples ({features.shape[0]})\n"
                    + f"{pearson_string}"
                )
                ax.set(title=title)

            fig = ax.figure
            plt.tight_layout()
            plt.savefig(filename)
            self.util.debug(f"saved regplot to {filename}")
            fig.clear()
            plt.close(fig)


    def plot_tree(self, model, features):
        from sklearn import tree

        ax = plt.gca()
        ax.figure.set_size_inches(100, 60)
        #        tree.plot_tree(model, ax = ax)
        tree.plot_tree(model, feature_names=list(features.columns), ax=ax)
        # plt.tight_layout()
        # print(ax)
        # one up because of the runs
        fig_dir = os.path.join(self.util.get_path("fig_dir"), "..")
        exp_name = self.util.get_exp_name(only_data=True)
        filename = os.path.join(fig_dir, f"{exp_name}EXPL_tree-plot.{self.format}")
        fig = ax.figure
        fig.savefig(filename)
        fig.clear()
        plt.close(fig)
        glob_conf.report.add_item(
            ReportItem(
                Header.HEADER_EXPLORE,
                "Tree plot",
                "for feature importance",
                filename,
            )
        )
