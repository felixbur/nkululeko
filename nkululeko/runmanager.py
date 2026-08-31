"""Runmanager module.

This module contains the Runmanager class which is responsible for managing the
runs of the experiment.
"""

from nkululeko.experiment_context import (
    ContextAware,
    bind_experiment_context,
    use_context,
)
from nkululeko.modelrunner import Modelrunner
from nkululeko.reporting.reporter import Reporter
from nkululeko.utils.util import Util


@bind_experiment_context
class Runmanager(ContextAware):
    """Class to manage the runs of the experiment (e.g. when results differ caused by random initialization)."""

    model = None  # The underlying model
    df_train, df_test, feats_train, feats_test = (
        None,
        None,
        None,
        None,
    )  # The dataframes
    reports = []

    def __init__(
        self,
        df_train,
        df_test,
        feats_train,
        feats_test,
        dev_x=None,
        dev_y=None,
        context=None,
    ):
        """Constructor setting up the dataframes.

        Args:
            df_train: train dataframe
            df_test: test dataframe
            feats_train: train features
            feats_train: test features

        """
        self.df_train, self.df_test, self.feats_train, self.feats_test = (
            df_train,
            df_test,
            feats_train,
            feats_test,
        )
        self.df_dev, self.feats_dev = dev_x, dev_y
        self.util = Util("runmanager", context=context)
        self.context = self.util.context
        self.target = self.context.config["DATA"]["target"]
        self.split3 = eval(self.util.config_val("EXP", "traindevtest", "False"))

    def do_runs(self):
        """Start the runs."""
        self.best_results = []  # keep the best result per run
        self.last_epochs = []  # keep the epoch of best result per run
        # for all runs
        for run in range(int(self.util.config_val("EXP", "runs", 1))):
            self.util.debug(
                f"run {run} using model {self.context.config['MODEL']['type']}"
            )
            # set the run index as global variable for reporting
            self.util.set_config_val("EXP", "run", run)
            if self.df_dev is not None:
                with use_context(self.context):
                    self.modelrunner = Modelrunner(
                        self.df_train,
                        self.df_dev,
                        self.feats_train,
                        self.feats_dev,
                        run,
                        split_name="dev",
                        context=self.context,
                    )
                self.reports, last_epoch = self.modelrunner.do_epochs()
            else:
                with use_context(self.context):
                    self.modelrunner = Modelrunner(
                        self.df_train,
                        self.df_test,
                        self.feats_train,
                        self.feats_test,
                        run,
                        split_name="test",
                        context=self.context,
                    )
                self.reports, last_epoch = self.modelrunner.do_epochs()

            last_report = self.reports[-1]
            plot_name_suggest = self.util.get_exp_name()
            plot_name = (
                self.util.config_val("PLOT", "name", plot_name_suggest)
                + f"_last_{last_report.run}_{last_report.epoch:03d}"
            )
            # finally, print out the numbers for this run
            self.print_report(last_report, plot_name)

            # wrap up the run
            plot_anim_progression = self.util.config_val("PLOT", "anim_progression", 0)
            if plot_anim_progression:
                plot_name_suggest = self.util.get_exp_name()
                plot_name = (
                    self.util.config_val("PLOT", "name", plot_name_suggest)
                    + "_conf_anim.gif"
                )
                self.util.debug(f"plotting animated confusion to {plot_name}")
                self.reports[-1].make_conf_animation(plot_name)
            plot_epoch_progression = self.util.config_val(
                "PLOT", "epoch_progression", 0
            )
            try:
                epoch_num = int(self.context.config["EXP"]["epochs"])
            except KeyError:
                # possibly this value has not been set
                epoch_num = 1
            if epoch_num > 1 and plot_epoch_progression:
                plot_name = (
                    self.util.config_val("PLOT", "name", plot_name_suggest)
                    + "_epoch_progression"
                )
                self.util.debug(f"plotting progression to {plot_name}")
                self.reports[-1].plot_epoch_progression(self.reports, plot_name)
            # remember the best run
            best_report = self.get_best_result(self.reports)
            plot_name = (
                self.util.config_val("PLOT", "name", plot_name_suggest)
                + f"_BEST-dev_{best_report.run}_{best_report.epoch:03d}"
            )
            # finally, print out the numbers for this run
            # self.print_report(best_report, plot_name)
            # remember the best run
            # Only print if best_report is not the same as last_report
            if best_report is not last_report:
                plot_name = (
                    self.util.config_val("PLOT", "name", plot_name_suggest)
                    + f"_BEST-dev_{best_report.run}_{best_report.epoch:03d}"
                )
                self.print_report(best_report, plot_name)

            self.best_results.append(best_report)
            self.last_epochs.append(last_epoch)
            if self.split3:
                # Load this run's own best checkpoint via best_report (scoped
                # to self.reports, this run only) - NOT get_best_model(),
                # which searches self.best_results across every run seen so
                # far. Calling get_best_model() here let an earlier run's
                # checkpoint silently win and get evaluated/reported as if it
                # were the current run's test result whenever that earlier
                # run's stored result outscored this run's own.
                best_model = self.load_model(best_report)
                self.test_report = self.modelrunner.eval_specific_model(
                    best_model, self.df_test, self.feats_test, split_name="test"
                )
                self.test_report.epoch = best_report.epoch
                plot_name = (
                    self.util.config_val("PLOT", "name", plot_name_suggest)
                    + f"_test_{best_report.run}_{best_report.epoch:03d}"
                )
                self.print_report(self.test_report, plot_name)
                # Replace the dev-phase result just appended above with the
                # real test result for top-level reporting - experiment.py
                # aggregates via self.best_results, so leaving the dev-based
                # best_report there would silently report dev performance
                # as if it were the final test score.
                self.best_results[-1] = self.test_report

    def print_best_result_runs(self):
        """Print the best result for all runs."""
        best_report = self.get_best_result(self.best_results)
        formatted_result = f"{best_report.result.test:.4f}"
        measure = self.util.config_val("MODEL", "measure", "uar").upper()
        # Determine split label based on whether we're using train/dev/test
        split_label = (
            f" ({self.modelrunner.split_name})"
            if hasattr(self, "modelrunner") and hasattr(self.modelrunner, "split_name")
            else ""
        )
        self.util.debug(
            f"best{split_label} result all runs with run {best_report.run} and epoch {best_report.epoch} with metric {measure}: {formatted_result}"
        )
        plot_name_suggest = self.util.get_exp_name()
        plot_name = (
            self.util.config_val("PLOT", "name", plot_name_suggest)
            + f"_BEST_{best_report.run}_{best_report.epoch:03d}_BEST_cnf"
        )
        self.print_report(best_report, plot_name)

    def print_given_result(self, run, epoch):
        """Print a result (confusion matrix) for a given epoch and run.

        Args:
            run: for which run
            epoch: for which epoch

        """
        report = Reporter([], [], run, epoch, context=self.context)
        self.util.debug(f"Re-testing result with run {run} and epoch {epoch}")
        plot_name_suggest = self.util.get_exp_name()
        plot_name = (
            self.util.config_val("PLOT", "name", plot_name_suggest)
            + f"_extra_{run}_{epoch:03d}_cnf"
        )
        self.print_report(report, plot_name)

    def print_report(self, report, plot_name):
        """Print a confusion matrix for a special report.

        Args:
            report: for which report (will be computed newly from model)
            plot_name: name of plot file
        """
        # self.load_model(report)
        # report = self.model.predict()
        self.util.debug(f"plotting conf matrix as {plot_name}")
        report.plot_confmatrix(plot_name, epoch=report.epoch)
        report.print_results(report.epoch, file_name=plot_name)
        report.print_probabilities(file_name=plot_name)

    def load_model(self, report):
        """Load a model from disk for a specific run and epoch and evaluate it.

        Args:
            report: for which report (will be re-evaluated)

        """
        run = report.run
        epoch = report.epoch
        self.util.set_config_val("EXP", "run", run)
        model_type = self.context.config["MODEL"]["type"]
        model = self.modelrunner._select_model(model_type)
        model.load(run, epoch)
        return model

    def get_best_model(self):
        best_report = self.get_best_result(self.best_results)
        return self.load_model(best_report)

    def get_best_result(self, reports):
        best_r = Reporter([], [], None, 0, 0, context=self.context)
        if self.util.high_is_good():
            best_r = self.search_best_result(reports, "ascending")
        else:
            best_r = self.search_best_result(reports, "descending")
        return best_r

    def search_best_result(self, reports, order):
        best_r = Reporter([], [], None, 0, 0, context=self.context)
        if order == "ascending":
            best_result = 0
            for r in reports:
                res = r.result.test
                if res > best_result:
                    best_result = res
                    best_r = r
        else:
            best_result = 10000
            for r in reports:
                res = r.result.test
                if res < best_result:
                    best_result = res
                    best_r = r
        formatted_result = f"{best_result:.4f}"
        # Determine split label based on whether we're using train/dev/test
        split_label = (
            f" ({self.modelrunner.split_name})"
            if hasattr(self, "modelrunner") and hasattr(self.modelrunner, "split_name")
            else ""
        )
        self.util.debug(
            f"search_best_result: order={order}, best epoch={best_r.epoch}, best{split_label} result={formatted_result}"
        )
        return best_r
