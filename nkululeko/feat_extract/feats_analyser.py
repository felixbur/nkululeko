# feats_analyser.py
import ast
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor
from nkululeko.util import Util
from nkululeko.plots import Plots
import nkululeko.glob_conf as glob_conf
from nkululeko.reporting.report_item import ReportItem
from nkululeko.reporting.defines import Header


class FeatureAnalyser:
    def __init__(self, label, df_labels, df_features):
        self.util = Util("feats_analyser")
        self.target = self.util.config_val("DATA", "target", "emotion")
        self.labels = df_labels[self.target]
        # self.labels = df_labels["class_label"]
        self.df_labels = df_labels
        self.features = df_features
        self.label = label

    def analyse(self):
        models = ast.literal_eval(self.util.config_val("EXPL", "model", "[log_reg]"))
        max_feat_num = int(self.util.config_val("EXPL", "max_feats", "10"))
        importance = None
        self.util.debug("analysing features...")
        result_importances = {}
        if self.util.exp_is_classification():
            for model_s in models:
                if model_s == "log_reg":
                    model = LogisticRegression()
                    model.fit(self.features, self.labels)
                    importance = model.coef_[0]
                    result_importances[model_s] = importance
                elif model_s == "tree":
                    model = DecisionTreeClassifier()
                    model.fit(self.features, self.labels)
                    importance = model.feature_importances_
                    result_importances[model_s] = importance
                    plot_tree = eval(self.util.config_val("EXPL", "plot_tree", "False"))
                    if plot_tree:
                        plots = Plots()
                        plots.plot_tree(model, self.features)
                elif model_s == "xgb":
                    model = XGBClassifier(enable_categorical=True, tree_method="hist")
                    self.labels = self.labels.astype("category")
                    model.fit(self.features, self.labels)
                    importance = model.feature_importances_
                    result_importances[model_s] = importance
                else:
                    self.util.error(f"invalid analysis method: {model}")
        else:  # regression experiment
            for model_s in models:
                if model_s == "lin_reg":
                    model = LinearRegression()
                    model.fit(self.features, self.labels)
                    importance = model.coef_
                    result_importances[model_s] = importance
                elif model_s == "tree":
                    model = DecisionTreeRegressor()
                    model.fit(self.features, self.labels)
                    importance = model.feature_importances_
                    result_importances[model_s] = importance
                elif model_s == "xgb":
                    model = XGBRegressor()
                    model.fit(self.features, self.labels)
                    importance = model.feature_importances_
                    result_importances[model_s] = importance
                else:
                    self.util.error(f"invalid analysis method: {model_s}")
        df_imp = pd.DataFrame(
            {
                "feats": self.features.columns,
            }
        )
        for model_s in result_importances:
            df_imp[f"{model_s}_importance"] = result_importances[model_s]
        df_imp["importance"] = df_imp.iloc[:, 1:].mean(axis=1).values
        df_imp = df_imp.sort_values(by="importance", ascending=False).iloc[
            :max_feat_num
        ]
        df_imp["importance"] = df_imp["importance"].map(
            lambda x: int(x * 1000) / 1000.0
        )
        ax = df_imp.plot(x="feats", y="importance", kind="bar")
        for p in ax.patches:
            ax.annotate(
                str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005)
            )
        ax.set(title=f"{self.label} samples")
        plt.tight_layout()
        fig_dir = self.util.get_path("fig_dir") + "../"  # one up because of the runs
        exp_name = self.util.get_exp_name(only_data=True)
        format = self.util.config_val("PLOT", "format", "png")
        model_name = "_".join(result_importances.keys())
        filename = f"{fig_dir}{exp_name}_EXPL_{model_name}.{format}"
        plt.savefig(filename)
        fig = ax.figure
        fig.clear()
        plt.close(fig)
        glob_conf.report.add_item(
            ReportItem(
                Header.HEADER_EXPLORE,
                f"Feature importance",
                f"using {model_name} models",
                filename,
            )
        )

        # result file
        res_dir = self.util.get_path("res_dir")
        file_name = (
            f"{res_dir}{self.util.get_exp_name(only_data=True)}EXPL_{model_s}.txt"
        )
        with open(file_name, "w") as text_file:
            text_file.write(
                "features in order of decreasing importance according to model"
                f" {model_s}:\n" + f"{str(df_imp.feats.values)}\n"
            )

        df_imp.to_csv(file_name, mode="a")

        # check if feature distributions should be plotted
        plot_feats = self.util.config_val("EXPL", "feature_distributions", False)
        if plot_feats:
            sample_selection = self.util.config_val("EXPL", "sample_selection", "all")
            for feature in df_imp.feats:
                # plot_feature(self, title, feature, label, df_labels, df_features):
                _plots = Plots()
                _plots.plot_feature(
                    sample_selection,
                    feature,
                    "class_label",
                    self.df_labels,
                    self.features,
                )
