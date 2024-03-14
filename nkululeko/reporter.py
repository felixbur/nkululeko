import ast
import glob
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    recall_score,
)
from sklearn.utils import resample

import nkululeko.glob_conf as glob_conf
from nkululeko.reporting.report_item import ReportItem
from nkululeko.result import Result
from nkululeko.reporting.defines import Header
from nkululeko.utils.util import Util


class Reporter:
    def __set_measure(self):
        if self.util.exp_is_classification():
            self.MEASURE = "UAR"
            self.result.measure = self.MEASURE
            self.is_classification = True
        else:
            self.is_classification = False
            self.measure = self.util.config_val("MODEL", "measure", "mse")
            if self.measure == "mse":
                self.MEASURE = "MSE"
                self.result.measure = self.MEASURE
            elif self.measure == "mae":
                self.MEASURE = "MAE"
                self.result.measure = self.MEASURE
            elif self.measure == "ccc":
                self.MEASURE = "CCC"
                self.result.measure = self.MEASURE

    def __init__(self, truths, preds, run, epoch):
        """Initialization with ground truth und predictions vector"""
        self.util = Util("reporter")
        self.format = self.util.config_val("PLOT", "format", "png")
        self.truths = truths
        self.preds = preds
        self.result = Result(0, 0, 0, 0, "unknown")
        self.run = run
        self.epoch = epoch
        self.__set_measure()
        self.cont_to_cat = False
        if len(self.truths) > 0 and len(self.preds) > 0:
            if self.util.exp_is_classification():
                self.result.test = recall_score(
                    self.truths, self.preds, average="macro"
                )
                self.result.loss = 1 - accuracy_score(self.truths, self.preds)
            else:
                # regression experiment
                if self.measure == "mse":
                    self.result.test = mean_squared_error(self.truths, self.preds)
                elif self.measure == "mae":
                    self.result.test = mean_absolute_error(self.truths, self.preds)
                elif self.measure == "ccc":
                    self.result.test = self.ccc(self.truths, self.preds)
                    if math.isnan(self.result.test):
                        self.util.debug(f"Truth: {self.truths}")
                        self.util.debug(f"Predict.: {self.preds}")
                        self.util.debug(f"Result is NAN: setting to -1")
                        self.result.test = -1
                else:
                    self.util.error(f"unknown measure: {self.measure}")

                # train and loss are being set by the model

    def set_id(self, run, epoch):
        """Make the report identifiable with run and epoch index"""
        self.run = run
        self.epoch = epoch

    def continuous_to_categorical(self):
        if self.cont_to_cat:
            return
        self.cont_to_cat = True
        bins = ast.literal_eval(glob_conf.config["DATA"]["bins"])
        self.truths = np.digitize(self.truths, bins) - 1
        self.preds = np.digitize(self.preds, bins) - 1

    def plot_confmatrix(self, plot_name, epoch):
        if not self.util.exp_is_classification():
            self.continuous_to_categorical()
        self._plot_confmat(self.truths, self.preds, plot_name, epoch)

    def plot_per_speaker(self, result_df, plot_name, function):
        """Plot a confusion matrix with the mode category per speakers
        Args:
            * result_df: a pandas dataframe with columns: preds, truths and speaker
        """
        speakers = result_df.speaker.unique()
        pred = np.zeros(0)
        truth = np.zeros(0)
        for s in speakers:
            s_df = result_df[result_df.speaker == s]
            mode = s_df.pred.mode().iloc[-1]
            mean = s_df.pred.mean()
            if function == "mode":
                s_df.pred = mode
            elif function == "mean":
                s_df.pred = mean
            else:
                self.util.error(f"unkown function {function}")
            pred = np.append(pred, s_df.pred.values)
            truth = np.append(truth, s_df["truth"].values)
        if not (self.is_classification or self.cont_to_cat):
            bins = ast.literal_eval(glob_conf.config["DATA"]["bins"])
            truth = np.digitize(truth, bins) - 1
            pred = np.digitize(pred, bins) - 1
        self._plot_confmat(truth, pred.astype("int"), plot_name, 0)

    def _plot_confmat(self, truths, preds, plot_name, epoch):
        # print(truths)
        # print(preds)
        fig_dir = self.util.get_path("fig_dir")
        labels = glob_conf.labels
        fig = plt.figure()  # figsize=[5, 5]
        uar = recall_score(truths, preds, average="macro")
        acc = accuracy_score(truths, preds)
        cm = confusion_matrix(
            truths, preds, normalize=None
        )  # normalize must be one of {'true', 'pred', 'all', None}
        if cm.shape[0] != len(labels):
            self.util.error(
                f"mismatch between confmatrix dim ({cm.shape[0]}) and labels"
                f" length ({len(labels)}: {labels})"
            )
        try:
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=labels
            ).plot(cmap="Blues")
        except ValueError:
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=list(labels).remove("neutral"),
            ).plot(cmap="Blues")

        reg_res = ""
        if not self.is_classification:
            reg_res = f", {self.MEASURE}: {self.result.test:.3f}"

        if epoch != 0:
            plt.title(f"Confusion Matrix, UAR: {uar:.3f}{reg_res}, Epoch: {epoch}")
        else:
            plt.title(f"Confusion Matrix, UAR: {uar:.3f}{reg_res}")
        img_path = f"{fig_dir}{plot_name}.{self.format}"
        plt.savefig(img_path)
        fig.clear()
        plt.close(fig)
        plt.savefig(img_path)
        plt.close(fig)
        glob_conf.report.add_item(
            ReportItem(
                Header.HEADER_RESULTS,
                self.util.get_model_description(),
                "Confusion matrix",
                img_path,
            )
        )

        res_dir = self.util.get_path("res_dir")
        uar = int(uar * 1000) / 1000.0
        acc = int(acc * 1000) / 1000.0
        rpt = f"epoch: {epoch}, UAR: {uar}, ACC: {acc}"
        # print(rpt)
        self.util.debug(rpt)
        file_name = f"{res_dir}{self.util.get_exp_name()}_conf.txt"
        with open(file_name, "w") as text_file:
            text_file.write(rpt)

    def print_results(self, epoch):
        """Print all evaluation values to text file"""
        res_dir = self.util.get_path("res_dir")
        file_name = f"{res_dir}{self.util.get_exp_name()}_{epoch}.txt"
        if self.util.exp_is_classification():
            labels = glob_conf.labels
            try:
                rpt = classification_report(
                    self.truths,
                    self.preds,
                    target_names=labels,
                    output_dict=True,
                )
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
                f1_per_class = f"result per class (F1 score): {c_ress}"
                self.util.debug(f1_per_class)
                rpt_str = f"{json.dumps(rpt)}\n{f1_per_class}"
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
            np.asarray(losses),
            np.asarray(results),
            np.asarray(train_results),
            np.asarray(losses_eval),
        )

        if np.all((results > 1)):
            # scale down values
            results = results / 100.0
            train_results = train_results / 100.0
        # if np.all((losses < 1)):
        # scale up values
        plt.figure(dpi=200)
        plt.plot(train_results, "green", label="train set")
        plt.plot(results, "red", label="dev set")
        plt.plot(losses, "black", label="losses")
        plt.plot(losses_eval, "grey", label="losses_eval")
        plt.xlabel("epochs")
        plt.ylabel(f"{self.MEASURE}")
        plt.legend()
        plt.savefig(f"{fig_dir}{out_name}.{self.format}")
        plt.close()

    @staticmethod
    def ccc(ground_truth, prediction):
        mean_gt = np.mean(ground_truth, 0)
        mean_pred = np.mean(prediction, 0)
        var_gt = np.var(ground_truth, 0)
        var_pred = np.var(prediction, 0)
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        cor = sum(v_pred * v_gt) / (np.sqrt(sum(v_pred**2)) * np.sqrt(sum(v_gt**2)))
        sd_gt = np.std(ground_truth)
        sd_pred = np.std(prediction)
        numerator = 2 * cor * sd_gt * sd_pred
        denominator = var_gt + var_pred + (mean_gt - mean_pred) ** 2
        ccc = numerator / denominator
        return ccc
