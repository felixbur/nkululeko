# feats_analyser.py
import ast

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import nkululeko.glob_conf as glob_conf
from nkululeko.plots import Plots
from nkululeko.reporting.defines import Header
from nkululeko.reporting.report_item import ReportItem
from nkululeko.utils.stats import normalize
from nkululeko.utils.util import Util


class FeatureAnalyser:
    def __init__(self, label, df_labels, df_features):
        self.util = Util("feats_analyser")
        self.target = self.util.config_val("DATA", "target", "emotion")
        self.labels = df_labels[self.target]
        # self.labels = df_labels["class_label"]
        self.df_labels = df_labels
        self.features = df_features
        self.label = label

    def _get_importance(self, model, permutation):
        model.fit(self.features, self.labels)
        if permutation:
            r = permutation_importance(
                model,
                self.features,
                self.labels,
                n_repeats=30,
                random_state=0,
            )
            importance = r["importances_mean"]
        else:
            importance = model.feature_importances_
        return importance

    def analyse_shap(self, model):
        """Shap analysis.

        Use the best model from a previous run and analyse feature importance with SHAP.
        https://m.mage.ai/how-to-interpret-and-explain-your-machine-learning-models-using-shap-values-471c2635b78e.
        """
        import shap

        name = "my_shap_values"
        if not self.util.exist_pickle(name):
            # get model name
            model_name = self.util.get_model_type()
            if hasattr(model, "predict_shap"):
                model_func = model.predict_shap
            elif hasattr(model, "clf"):
                model_func = model.clf.predict
            else:
                raise Exception("Model not supported for SHAP analysis")

            self.util.debug(f"using SHAP explainer for {model_name} model")

            explainer = shap.Explainer(
                model_func,
                self.features,
                output_names=glob_conf.labels,
                algorithm="permutation",
                npermutations=5,
            )

            self.util.debug("computing SHAP values...")
            shap_values = explainer(self.features)
            self.util.to_pickle(shap_values, name)
        else:
            shap_values = self.util.from_pickle(name)
        # plt.figure()
        plt.close("all")
        plt.tight_layout()
        shap.plots.bar(shap_values)
        fig_dir = self.util.get_path("fig_dir") + "../"  # one up because of the runs
        exp_name = self.util.get_exp_name(only_data=True)
        format = self.util.config_val("PLOT", "format", "png")
        filename = f"_SHAP_{model.name}"
        filename = f"{fig_dir}{exp_name}{filename}.{format}"
        plt.savefig(filename)
        plt.close()
        self.util.debug(f"plotted SHAP feature importance to {filename}")

    def analyse(self):
        models = ast.literal_eval(self.util.config_val("EXPL", "model", "['log_reg']"))
        model_name = "_".join(models)
        max_feat_num = int(self.util.config_val("EXPL", "max_feats", "10"))
        # https://scikit-learn.org/stable/modules/permutation_importance.html
        permutation = eval(self.util.config_val("EXPL", "permutation", "False"))
        importance = None
        self.util.debug("analysing features...")
        result_importances = {}
        if self.util.exp_is_classification():
            for model_s in models:
                if permutation:
                    self.util.debug(
                        f"computing feature importance via permutation for {model_s}, might take longer..."
                    )
                if model_s == "bayes":
                    from sklearn.naive_bayes import GaussianNB

                    model = GaussianNB()
                    result_importances[model_s] = self._get_importance(
                        model, permutation
                    )
                elif model_s == "gmm":
                    from sklearn import mixture

                    n_components = int(
                        self.util.config_val("MODEL", "GMM_components", "4")
                    )
                    covariance_type = self.util.config_val(
                        "MODEL", "GMM_covariance_type", "full"
                    )
                    model = mixture.GaussianMixture(
                        n_components=n_components, covariance_type=covariance_type
                    )
                    result_importances[model_s] = self._get_importance(
                        model, permutation
                    )
                elif model_s == "knn":
                    from sklearn.neighbors import KNeighborsClassifier

                    method = self.util.config_val("MODEL", "KNN_weights", "uniform")
                    k = int(self.util.config_val("MODEL", "K_val", "5"))
                    model = KNeighborsClassifier(
                        n_neighbors=k, weights=method
                    )  # set up the classifier
                    result_importances[model_s] = self._get_importance(
                        model, permutation
                    )
                elif model_s == "log_reg":
                    model = LogisticRegression()
                    model.fit(self.features, self.labels)
                    if permutation:
                        r = permutation_importance(
                            model,
                            self.features,
                            self.labels,
                            n_repeats=30,
                            random_state=0,
                        )
                        importance = r["importances_mean"]
                    else:
                        importance = model.coef_[0]
                    result_importances[model_s] = importance
                elif model_s == "svm":
                    from sklearn.svm import SVC

                    c = float(self.util.config_val("MODEL", "C_val", "1.0"))
                    model = SVC(kernel="linear", C=c, gamma="scale")
                    result_importances[model_s] = self._get_importance(
                        model, permutation
                    )
                    plot_tree = eval(self.util.config_val("EXPL", "plot_tree", "False"))
                    if plot_tree:
                        plots = Plots()
                        plots.plot_tree(model, self.features)
                elif model_s == "tree":
                    model = DecisionTreeClassifier()
                    result_importances[model_s] = self._get_importance(
                        model, permutation
                    )
                    plot_tree = eval(self.util.config_val("EXPL", "plot_tree", "False"))
                    if plot_tree:
                        plots = Plots()
                        plots.plot_tree(model, self.features)
                elif model_s == "xgb":
                    from xgboost import XGBClassifier

                    model = XGBClassifier(enable_categorical=True, tree_method="hist")
                    self.labels = self.labels.astype("category")
                    result_importances[model_s] = self._get_importance(
                        model, permutation
                    )
                else:
                    self.util.error(f"invalid analysis method: {model}")
        else:  # regression experiment
            for model_s in models:
                if permutation:
                    self.util.debug(
                        f"computing feature importance via permutation for {model_s}, might take longer..."
                    )
                if model_s == "knn_reg":
                    from sklearn.neighbors import KNeighborsRegressor

                    method = self.util.config_val("MODEL", "KNN_weights", "uniform")
                    k = int(self.util.config_val("MODEL", "K_val", "5"))
                    model = KNeighborsRegressor(
                        n_neighbors=k, weights=method
                    )  # set up the classifier
                    result_importances[model_s] = self._get_importance(
                        model, permutation
                    )
                elif model_s == "lin_reg":
                    model = LinearRegression()
                    model.fit(self.features, self.labels)
                    if permutation:
                        r = permutation_importance(
                            model,
                            self.features,
                            self.labels,
                            n_repeats=30,
                            random_state=0,
                        )
                        importance = r["importances_mean"]
                    else:
                        importance = model.coef_
                    result_importances[model_s] = importance
                elif model_s == "tree_reg":
                    model = DecisionTreeRegressor()
                    result_importances[model_s] = self._get_importance(
                        model, permutation
                    )
                elif model_s == "xgr":
                    from xgboost import XGBRegressor

                    model = XGBRegressor()
                    result_importances[model_s] = self._get_importance(
                        model, permutation
                    )
                else:
                    self.util.error(f"invalid analysis method: {model_s}")
        df_imp = pd.DataFrame(
            {
                "feats": self.features.columns,
            }
        )
        for model_s in result_importances:
            if len(result_importances) == 1:
                df_imp[f"{model_s}_importance"] = result_importances[model_s]
            else:
                # normalize the distributions because they might be different
                self.util.debug(f"scaling importance values for {model_s}")
                importance = result_importances[model_s]
                importance = normalize(importance.reshape(-1, 1))
                df_imp[f"{model_s}_importance"] = importance

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
        title = (
            f"Feature importance for {self.label} samples with model(s) {model_name}"
        )
        if permutation:
            title += "\n based on feature permutation"
        ax.set(title=title)
        plt.tight_layout()
        fig_dir = self.util.get_path("fig_dir") + "../"  # one up because of the runs
        exp_name = self.util.get_exp_name(only_data=True)
        format = self.util.config_val("PLOT", "format", "png")
        filename = f"_EXPL_{model_name}"
        if permutation:
            filename += "_perm"
        filename = f"{fig_dir}{exp_name}{filename}.{format}"
        plt.savefig(filename)
        fig = ax.figure
        fig.clear()
        plt.close(fig)
        caption = "Feature importance"
        if permutation:
            caption += " based on permutation of features."
        glob_conf.report.add_item(
            ReportItem(
                Header.HEADER_EXPLORE,
                caption,
                f"using {model_name} models",
                filename,
            )
        )

        # print feature importance values to file and debug and save to result
        self.util.debug(
            f"Importance features from {model_name}: features = \n{df_imp['feats'].values.tolist()}"
        )
        # result file
        res_dir = self.util.get_path("res_dir")
        filename = f"_EXPL_{model_name}"
        if permutation:
            filename += "_perm"
        filename = f"{res_dir}{self.util.get_exp_name(only_data=True)}{filename}_{max_feat_num}_fi.txt"
        with open(filename, "w") as text_file:
            text_file.write(
                "features in order of decreasing importance according to model"
                f" {model_name}:\n" + f"{str(df_imp.feats.values)}\n"
            )

        df_imp.to_csv(filename, mode="a")
        self.util.debug(f"Saved feature importance values to {filename}")

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
