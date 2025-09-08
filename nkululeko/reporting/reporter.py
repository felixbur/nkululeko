import ast
import glob
import json
import math
import os
import audplot

# import os
from confidence_intervals import evaluate_with_conf_int
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from scipy.stats import pearsonr
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score

# from torch import is_tensor
from audmetric import accuracy
from audmetric import concordance_cc
from audmetric import mean_absolute_error
from audmetric import mean_squared_error
from audmetric import unweighted_average_recall

import nkululeko.glob_conf as glob_conf
from nkululeko.plots import Plots
from nkululeko.reporting.defines import Header
from nkululeko.reporting.report_item import ReportItem
from nkululeko.reporting.result import Result
from nkululeko.utils.util import Util


class Reporter:
    def _set_metric(self):
        if self.util.exp_is_classification():
            self.metric = "uar"
            self.METRIC = "UAR"
            self.result.metric = self.METRIC
            self.is_classification = True
        else:
            self.is_classification = False
            self.metric = self.util.config_val("MODEL", "measure", "mse")
            if self.metric == "mse":
                self.METRIC = "MSE"
                self.result.metric = self.METRIC
            elif self.metric == "mae":
                self.METRIC = "MAE"
                self.result.metric = self.METRIC
            elif self.metric == "ccc":
                self.METRIC = "CCC"
                self.result.metric = self.METRIC

    def __init__(self, truths, preds, run, epoch, probas=None):
        """Initialization with ground truth und predictions vector.

        Args:
            truths (list): the ground truth
            preds (list): the predictions
            run (int): number of run
            epoch (int): number of epoch
            probas (pd.Dataframe, optional): probabilities per class. Defaults to None.
        """
        self.util = Util("reporter")
        self.probas = probas
        self.format = self.util.config_val("PLOT", "format", "png")
        self.truths = np.asarray(truths)
        self.preds = np.asarray(preds)
        self.result = Result(0, 0, 0, 0, "unknown")
        self.run = run
        self.epoch = epoch
        self.model_type = self.util.get_model_type()
        self._set_metric()
        self.filenameadd = ""
        self.cont_to_cat = False
        if len(self.truths) > 0 and len(self.preds) > 0:
            if self.util.exp_is_classification():
                uar, upper, lower = self._get_test_result(
                    self.truths, self.preds, "uar"
                )
                self.result.test = uar
                self.result.set_upper_lower(upper, lower)
                self.result.loss = 1 - accuracy(self.truths, self.preds)
            else:
                # regression experiment
                # keep the original values for further use, they will be binned later
                self.truths_cont = self.truths
                self.preds_cont = self.preds
                test_result, upper, lower = self._get_test_result(
                    self.truths, self.preds, self.metric
                )
                self.result.test = test_result
                self.result.set_upper_lower(upper, lower)
                # train and loss are being set by the model

    def _get_test_result(self, truths, preds, metric):
        if metric == "uar":
            test_result, (upper, lower) = evaluate_with_conf_int(
                preds,
                unweighted_average_recall,
                truths,
                num_bootstraps=1000,
                alpha=5,
            )
        elif metric == "mse":
            test_result, (upper, lower) = evaluate_with_conf_int(
                preds,
                mean_squared_error,
                truths,
                num_bootstraps=1000,
                alpha=5,
            )
        elif metric == "mae":
            test_result, (upper, lower) = evaluate_with_conf_int(
                preds,
                mean_absolute_error,
                truths,
                num_bootstraps=1000,
                alpha=5,
            )
        elif metric == "ccc":
            test_result, (upper, lower) = evaluate_with_conf_int(
                preds,
                concordance_cc,
                truths,
                num_bootstraps=1000,
                alpha=5,
            )
            if math.isnan(test_result):
                self.util.debug(f"Truth: {self.truths}")
                self.util.debug(f"Predict.: {self.preds}")
                self.util.debug("Result is NAN: setting to -1")
                test_result = -1
        else:
            self.util.error(f"unknown metric: {self.metric}")
        return test_result, upper, lower

    def print_probabilities(self, file_name=None):
        """Print the probabilities per class to a file in the store."""
        if (
            self.util.exp_is_classification()
            and self.probas is not None
            and "uncertainty" not in self.probas
        ):
            probas = self.probas
            # softmax the probabilities or logits
            uncertainty = probas.apply(softmax, axis=1)
            probas["predicted"] = self.preds
            probas["truth"] = self.truths
            try:
                le = glob_conf.label_encoder
                if le is not None:
                    mapping = dict(zip(le.classes_, range(len(le.classes_))))
                    mapping_reverse = {value: key for key, value in mapping.items()}
                    probas = probas.rename(columns=mapping_reverse)
                    probas["predicted"] = probas["predicted"].map(mapping_reverse)
                    probas["truth"] = probas["truth"].map(mapping_reverse)
                else:
                    self.util.debug("Label encoder is None, skipping label mapping")
            except AttributeError as ae:
                self.util.debug(f"Can't label categories: {ae}")
            # compute entropy per sample
            uncertainty = uncertainty.apply(entropy)
            # scale it to 0-1
            max_ent = math.log(len(glob_conf.labels))
            uncertainty = (uncertainty - uncertainty.min()) / (
                max_ent - uncertainty.min()
            )
            probas["uncertainty"] = uncertainty
            probas["correct"] = probas.predicted == probas.truth
            if file_name is None:
                file_name = self.util.get_pred_name() + ".csv"
            else:
                # Ensure the file_name goes to the results directory
                if not os.path.isabs(file_name):
                    res_dir = self.util.get_res_dir()
                    if not file_name.endswith(".csv"):
                        file_name = os.path.join(res_dir, file_name + ".csv")
                    else:
                        file_name = os.path.join(res_dir, file_name)
                else:
                    if not file_name.endswith(".csv"):
                        file_name = file_name + ".csv"
            self.probas = probas
            self.plot_proba_conf()
            probas.to_csv(file_name)
            self.util.debug(f"Saved probabilities to {file_name}")
            plots = Plots()
            ax, caption = plots.plotcatcont(
                probas, "correct", "uncertainty", "uncertainty", "correct"
            )
            file_name = f"{self.util.get_exp_name()}_uncertainty"
            plots.save_plot(
                ax,
                caption,
                "uncertainty",
                file_name,
                "samples",
            )

    def plot_proba_conf(self):
        uncertainty_threshold = self.util.config_val(
            "PLOT", "uncertainty_threshold", False
        )
        if uncertainty_threshold:
            uncertainty_threshold = float(uncertainty_threshold)
            old_size = self.probas.shape[0]
            df = self.probas[self.probas["uncertainty"] < uncertainty_threshold]
            new_size = df.shape[0]
            difference = old_size - new_size
            diff_perc = int((difference / old_size) * 100 if old_size > 0 else 0)
            self.util.debug(
                f"Filtered probabilities: {old_size} -> {new_size} ({difference}) samples with uncertainty < {uncertainty_threshold}"
            )
            truths = df["truth"].values
            preds = df["predicted"].values
            plot_name = f"{self.util.get_exp_name()}_uncertainty_lt_{uncertainty_threshold}_{diff_perc}pct-removed_cnf"
            self._plot_confmat(
                truths,
                preds,
                plot_name,
                epoch=None,
                test_result=None,
            )

    def set_id(self, run, epoch):
        """Make the report identifiable with run and epoch index."""
        self.run = run
        self.epoch = epoch

    def continuous_to_categorical(self):
        if self.cont_to_cat:
            return
        self.cont_to_cat = True
        bins = ast.literal_eval(glob_conf.config["DATA"]["bins"])
        self.truths = np.digitize(self.truths, bins) - 1
        self.preds = np.digitize(self.preds, bins) - 1

    def plot_confmatrix(self, plot_name, epoch=None):
        """Plot a confusionmatrix to the store.

        Args:
            plot_name (str): name for the image file.
            epoch (int, optional): Number of epoch. Defaults to None.
        """
        if not self.util.exp_is_classification():
            self._plot_scatter(self.truths, self.preds, f"{plot_name}_scatter", epoch)
            self.continuous_to_categorical()
        self._plot_confmat(self.truths, self.preds, plot_name, epoch)

    def plot_per_speaker(self, result_df, plot_name, function):
        """Plot a confusion matrix with the mode category per speakers.

        If the function is mode and the values continuous, bin first

        Args:
            result_df: a pandas dataframe with columns: preds, truths and speaker.
            plot_name: name for the figure.
            function: either mode or mean.
        """
        if function == "mode" and not self.is_classification:
            truths, preds = result_df["truths"].values, result_df["preds"].values
            truths, preds = self.util._bin_distributions(truths, preds)
            result_df["truths"], result_df["preds"] = truths, preds
        speakers = result_df.speakers.unique()
        preds_speakers = np.zeros(0)
        truths_speakers = np.zeros(0)
        for s in speakers:
            s_df = result_df[result_df.speakers == s]
            s_truth = s_df.truths.iloc[0]
            s_pred = None
            if function == "mode":
                s_pred = s_df.preds.mode().iloc[-1]
            elif function == "mean":
                s_pred = s_df.preds.mean()
            else:
                self.util.error(f"unknown function {function}")
            preds_speakers = np.append(preds_speakers, s_pred)
            truths_speakers = np.append(truths_speakers, s_truth)
        test_result, upper, lower = self._get_test_result(
            result_df.truths.values, result_df.preds.values, self.metric
        )
        test_result = Result(test_result, None, None, None, self.METRIC)
        test_result.set_upper_lower(upper, lower)
        result_msg = f"Speaker combination result: {test_result.test_result_str()}"
        self.util.debug(result_msg)
        if function == "mean":
            truths_speakers, preds_speakers = self.util._bin_distributions(
                truths_speakers, preds_speakers
            )
        self._plot_confmat(
            truths_speakers,
            preds_speakers.astype("int"),
            plot_name,
            test_result=test_result,
        )

    def _plot_scatter(self, truths, preds, plot_name, epoch=None):
        # print(truths)
        # print(preds)
        if epoch is None:
            epoch = self.epoch
        fig_dir = self.util.get_path("fig_dir")
        pcc = pearsonr(self.truths, self.preds)[0]
        reg_res = self.result.test_result_str()
        fig = plt.figure()
        plt.scatter(truths, preds)
        plt.xlabel("truth")
        plt.ylabel("prediction")

        if epoch != 0:
            plt.title(f"Scatter plot: {reg_res}, PCC: {pcc:.3f}, Epoch: {epoch}")
        else:
            plt.title(f"Scatter plot: {reg_res}, PCC: {pcc:.3f}")

        img_path = f"{fig_dir}{plot_name}{self.filenameadd}.{self.format}"
        plt.savefig(img_path)
        self.util.debug(f"Saved scatter plot to {img_path}")
        fig.clear()
        plt.close(fig)
        plt.close()
        plt.clf()
        glob_conf.report.add_item(
            ReportItem(
                Header.HEADER_RESULTS,
                self.util.get_model_description(),
                "Result scatter plot",
                img_path,
            )
        )

    def _plot_confmat(self, truths, preds, plot_name, epoch=None, test_result=None):
        if epoch is None:
            epoch = self.epoch
        if test_result is None:
            test_result = self.result
        fig_dir = self.util.get_path("fig_dir")
        fig = plt.figure()  # figsize=[5, 5]
        uar, (upper, lower) = evaluate_with_conf_int(
            preds,
            unweighted_average_recall,
            truths,
            num_bootstraps=1000,
            alpha=5,
        )
        acc = accuracy(truths, preds)
        #         display_labels=list(labels).remove("neutral"),
        #     ).plot(cmap="Blues")
        # Check if percentages should be shown in confusion matrix
        show_percentages = self.util.config_val("PLOT", "show_percentages", True)
        
        le = glob_conf.label_encoder
        if le is not None:
            label_dict = dict(zip(range(len(le.classes_)), le.classes_))
            audplot.confusion_matrix(
                truths, preds, label_aliases=label_dict, show_both=show_percentages
            )
        else:
            audplot.confusion_matrix(truths, preds, show_both=show_percentages)
        reg_res = ""
        if not self.is_classification:
            reg_res = f"{test_result.test_result_str()}"
            self.util.debug(
                f"Best result at epoch {epoch}: {test_result.test_result_str()}"
            )

        uar_str = self.util.to_3_digits_str(uar)
        acc_str = self.util.to_3_digits_str(acc)
        up_str = self.util.to_3_digits_str(upper)
        low_str = self.util.to_3_digits_str(lower)

        if epoch != 0:
            plt.title(
                f"Confusion Matrix, UAR: {uar_str} "
                + f"(+-{up_str}/{low_str}), {reg_res}, Epoch: {epoch}"
            )
        else:
            plt.title(
                f"Confusion Matrix, UAR: {uar_str} "
                + f"(+-{up_str}/{low_str}) {reg_res}"
            )
        img_path = f"{fig_dir}{plot_name}{self.filenameadd}.{self.format}"
        plt.savefig(img_path)
        self.util.debug(f"Saved confusion plot to {img_path}")
        fig.clear()
        plt.close(fig)
        plt.close()
        plt.clf()
        glob_conf.report.add_item(
            ReportItem(
                Header.HEADER_RESULTS,
                self.util.get_model_description(),
                "Confusion matrix",
                img_path,
            )
        )

        res_dir = self.util.get_path("res_dir")
        rpt = (
            f"Confusion matrix result for epoch: {epoch}, UAR: {uar_str}"
            + f", (+-{up_str}/{low_str}), ACC: {acc_str}"
        )
        # print(rpt)
        self.util.debug(rpt)
        file_name = f"{res_dir}{self.util.get_exp_name()}{self.filenameadd}_conf.txt"
        with open(file_name, "w") as text_file:
            text_file.write(rpt)

    def set_filename_add(self, my_string):
        self.filenameadd = f"_{my_string}"

    def print_logo(self, results):
        res_dir = self.util.get_path("res_dir")
        result_str = f"LOGO results: [{','.join(results.astype(str))}]"
        file_name = f"{res_dir}/logo_results.txt"
        with open(file_name, "w") as text_file:
            text_file.write(
                f"LOGO: mean {results.mean():.3f},  std: " + f"{results.std():.3f}"
            )
            text_file.write("\n")
            text_file.write(result_str)
        self.util.debug(result_str)

    def print_results(self, epoch=None, file_name=None):
        if epoch is None:
            epoch = self.epoch
        """Print all evaluation values to text file."""
        res_dir = self.util.get_path("res_dir")
        if file_name is None:
            file_name = (
                f"{res_dir}{self.util.get_exp_name()}_{epoch}{self.filenameadd}.txt"
            )
        else:
            self.util.debug(f"####->{file_name}<-####")
            file_name = f"{res_dir}{file_name}{self.filenameadd}.txt"
        if self.util.exp_is_classification():
            if glob_conf.label_encoder is not None:
                labels = glob_conf.label_encoder.classes_
            else:
                labels = glob_conf.labels
            try:
                rpt = classification_report(
                    self.truths,
                    self.preds,
                    target_names=labels,
                    output_dict=True,
                )
                # in case we hade numbers before
                s_labels = [str(x) for x in labels]
                # print classifcation report in console
                class_report_str = classification_report(
                    self.truths,
                    self.preds,
                    target_names=s_labels,
                    digits=4,
                )
                self.util.debug(f"\n {class_report_str}")
            except ValueError as e:
                self.util.debug(
                    "Reporter: caught a ValueError when trying to get"
                    " classification_report: " + e
                )
                rpt = self.result.to_string()
            with open(file_name, "w") as text_file:
                c_ress = list(range(len(labels)))
                for i, l in enumerate(labels):
                    c_res = rpt[l]["f1-score"]
                    c_ress[i] = float(f"{c_res:.3f}")
                self.util.debug(f"labels: {labels}")
                f1_per_class = (
                    f"result per class (F1 score): {c_ress} from epoch: {epoch}"
                )
                self.util.debug(f1_per_class)
                # convert all keys to strings
                rpt = dict((str(key), value) for (key, value) in rpt.items())
                rpt_str = f"{json.dumps(rpt)}\n{f1_per_class}"
                # rpt_str += f"\n{auc_auc}"
                text_file.write(rpt_str)
                glob_conf.report.add_item(
                    ReportItem(
                        Header.HEADER_RESULTS,
                        f"Classification result {self.util.get_model_description()}",
                        rpt_str,
                    )
                )

        else:  # regression
            result = self.result.test
            r2 = r2_score(self.truths, self.preds)
            pcc = pearsonr(self.truths, self.preds)[0]
            measure = self.util.config_val("MODEL", "measure", "mse")
            with open(file_name, "w") as text_file:
                text_file.write(
                    f"{measure}: {result:.3f}, r_2: {r2:.3f}, pcc {pcc:.3f}"
                )

    def make_conf_animation(self, out_name):
        import imageio

        fig_dir = self.util.get_path("fig_dir")
        filenames = glob.glob(fig_dir + f"{self.util.get_plot_name()}*_?_???_cnf.png")
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        fps = self.util.config_val("PLOT", "fps", "1")
        try:
            imageio.mimsave(fig_dir + out_name, images, fps=int(fps))
        except RuntimeError as e:
            self.util.error("error writing anim gif: " + e)

    def get_result(self):
        return self.result

    def plot_epoch_progression_finetuned(self, df):
        plot_name_suggest = self.util.get_exp_name()
        fig_dir = self.util.get_path("fig_dir")
        plot_name = (
            self.util.config_val("PLOT", "name", plot_name_suggest)
            + "_epoch_progression"
        )
        ax = df.plot()
        fig = ax.figure
        plt.xlabel("epochs")
        plt.ylabel(f"{self.METRIC}")
        plot_path = f"{fig_dir}{plot_name}.{self.format}"
        plt.savefig(plot_path)
        self.util.debug(f"plotted epoch progression to {plot_path}")
        plt.close(fig)
        # fig.clear()

    def plot_epoch_progression(self, reports, out_name):
        fig_dir = self.util.get_path("fig_dir")
        results, losses, train_results, losses_eval = [], [], [], []
        for r in reports:
            results.append(r.get_result().test)
            losses.append(r.get_result().loss)
            train_results.append(r.get_result().train)
            losses_eval.append(r.get_result().loss_eval)

        # do a plot per run
        # scale the losses so they fit on the picture
        losses, results, train_results, losses_eval = (
            self._scaleresults(np.asarray(losses)),
            self._scaleresults(np.asarray(results)),
            self._scaleresults(np.asarray(train_results)),
            self._scaleresults(np.asarray(losses_eval)),
        )

        plt.figure(dpi=200)
        plt.plot(train_results, "green", label="train set")
        plt.plot(results, "red", label="dev set")
        plt.plot(losses, "black", label="losses")
        plt.plot(losses_eval, "grey", label="losses_eval")
        plt.xlabel("epochs")
        plt.ylabel(f"{self.METRIC}")
        plt.legend()
        plt.savefig(f"{fig_dir}{out_name}.{self.format}")
        plt.close()

    def _scaleresults(self, results: np.ndarray) -> np.ndarray:
        results = results.copy()
        """Scale results to fit on the plot."""
        if np.any((results > 1)):
            # scale down values
            results = results / 100.0
        return results
