# runmanager.py

from model_svm import SVM_model
from model_svr import SVR_model
from model_xgb import XGB_model
from model_xgr import XGR_model
from model_mlp import MLP_model
from model_mlp_regression import MLP_Reg_model
from reporter import Reporter
from result import Result
import ast
from util import Util  
import glob_conf

class Runmanager:
    """Class to manage the runs of the experiment (e.g. when results differ caused by random initialization)"""
    model = None  # The underlying model
    df_train, df_test, feats_train, feats_test = None, None, None, None # The dataframes
    reports = []


    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor setting up the dataframes
        Args:
            df_train: train dataframe
            df_test: test dataframe
            feats_train: train features
            feats_train: test features

        Returns:
        
        """
        self.df_train, self.df_test, self.feats_train, self.feats_test = df_train, df_test, feats_train, feats_test
        self.util = Util()
        self.target = glob_conf.config['DATA']['target']
        # intialize a new model
        model_type = glob_conf.config['MODEL']['type']
        if model_type=='svm':
            self.model = SVM_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
        elif model_type=='svr':
            self.model = SVR_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
        elif model_type=='xgb':
            self.model = XGB_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
        elif model_type=='xgr':
            self.model = XGR_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
        elif model_type=='cnn':
            from model_cnn import CNN_model
            self.model = CNN_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
        elif model_type=='mlp':
            self.model = MLP_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
        elif model_type=='mlp_reg':
            self.model = MLP_Reg_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
        else:
            self.util.error(f'unknown model type: \'{model_type}\'')


    def do_runs(self):
        """Start the runs"""
        self.best_results = [] # keep the best result per run 
        # for all runs
        for run in range(int(self.util.config_val('EXP', 'runs', 1))):
            self.util.debug(f'run {run}')
            # set the run index as global variable for reporting
            self.util.set_config_val('EXP', 'run', run)
            # initialze results
            self.reports = []
            plot_epochs = self.util.config_val('PLOT', 'epochs', False)
            only_test = self.util.config_val('MODEL', 'only_test', False)
            # for all epochs
            for epoch in range(int(self.util.config_val('EXP', 'epochs', 1))):
                if only_test:
                    self.model.load(run, epoch)
                    self.util.debug(f'reusing model: {self.model.store_path}')
                    self.model.reset_test(self.df_test, self.feats_test)
                else:
                    self.model.set_id(run, epoch)
                    self.model.train()
                report = self.model.predict()
                report.set_id(run, epoch)
                plot_name = self.util.get_plot_name()+f'_{run}_{epoch:03d}_cnf.png'
                self.reports.append(report)                
                self.util.debug(f'run: {run} epoch: {epoch}: result: '
                    f'{self.reports[-1].get_result().get_test_result()}')
                if plot_epochs:
                    self.util.debug(f'plotting conf matrix to {plot_name}')
                    report.plot_confmatrix(plot_name, epoch)
                store_models = self.util.config_val('MODEL', 'save', 0)
                plot_best_model = self.util.config_val('PLOT', 'best_model', False)
                if (store_models or plot_best_model) and (not only_test): # in any case the model needs to be stored to disk.
                    self.model.store()
            if not plot_epochs:
                # Do a final confusion matrix plot
                self.util.debug(f'plotting final conf matrix to {plot_name}')
                self.reports[-1].plot_confmatrix(plot_name, epoch)
            # wrap up the run 
            plot_anim_progression = self.util.config_val('PLOT', 'anim_progression', 0)
            if plot_anim_progression:
                plot_name_suggest = self.util.get_exp_name()
                plot_name = self.util.config_val('PLOT', 'name', plot_name_suggest)+'_conf_anim.gif'
                self.util.debug(f'plotting animated confusion to {plot_name}')
                self.reports[-1].make_conf_animation(plot_name)
            plot_epoch_progression = self.util.config_val('PLOT', 'epoch_progression', 0)
            if plot_epoch_progression:
                plot_name_suggest = self.util.get_exp_name()
                plot_name = self.util.config_val('PLOT', 'name', plot_name_suggest)+'_epoch_progression.png'
                self.util.debug(f'plotting progression to {plot_name}')
                self.reports[-1].plot_epoch_progression(self.reports, plot_name)
            # remember the best run
            best_report = self.get_best_result(self.reports)
            plot_best_model = self.util.config_val('PLOT', 'best_model', False)
            if plot_best_model:
                plot_name_suggest = self.util.get_exp_name()
                plot_name = self.util.config_val('PLOT', 'name', plot_name_suggest)+f'_BEST_{best_report.run}_{best_report.epoch:03d}_BEST_cnf.png'
                self.util.debug(f'best result with run {best_report.run} and epoch {best_report.epoch}: {best_report.result.get_test_result()}')
                self.print_model(best_report, plot_name)
            # finally, print out the numbers for this run
            self.reports[-1].print_results(int(self.util.config_val('EXP', 'epochs', 1)))
            self.best_results.append(best_report)

    def print_best_result_runs(self):
        """Print the best result for all runs"""
        best_report = self.get_best_result(self.best_results)
        self.util.debug(f'best result all runs with run {best_report.run} \
            and epoch {best_report.epoch}: {best_report.result.test:.3f}')
        plot_name_suggest = self.util.get_exp_name()
        plot_name = self.util.config_val('PLOT', 'name', plot_name_suggest)+f'_BEST_{best_report.run}_{best_report.epoch:03d}_BEST_cnf.png'
        self.print_model(best_report, plot_name)

    def print_given_result(self, run, epoch):
        """Print a result (confusion matrix) for a given epoch and run
        Args: 
            run: for which run
            epoch: for which epoch
        
        """
        report =  Reporter([], [])
        report.set_id(run, epoch)
        self.util.debug(f'Re-testing result with run {run} and epoch {epoch}')
        plot_name_suggest = self.util.get_exp_name()
        plot_name = self.util.config_val('PLOT', 'name', plot_name_suggest)+f'_extra_{run}_{epoch:03d}_cnf.png'
        self.print_model(report, plot_name)

    def print_model(self, report, plot_name):
        """Print a confusion matrix for a special report
        Args:
            report: for which report (will be computed newly from model)
            plot_name: name of plot file
        """
        epoch = report.epoch
        # self.load_model(report)
        # report = self.model.predict()
        self.util.debug(f'plotting conf matrix to {plot_name}')
        report.plot_confmatrix(plot_name, epoch)
        report.print_results(epoch)


    def load_model(self, report):
        """Load a model from disk for a specific run and epoch and evaluate
        Args:
            report: for which report (will be re-evaluated)
        
        """
        run = report.run
        epoch = report.epoch
        self.util.set_config_val('EXP', 'run', run)
        model_type = glob_conf.config['MODEL']['type']
        if model_type=='svm':
            self.model = SVM_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
        elif model_type=='svr':
            self.model = SVR_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
        elif model_type=='xgb':
            self.model = XGB_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
        elif model_type=='xgr':
            self.model = XGR_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
        elif model_type=='cnn':
            from model_cnn import CNN_model
            self.model = CNN_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
        elif model_type=='mlp':
            self.model = MLP_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
        elif model_type=='mlp_reg':
            self.model = MLP_Reg_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
        else:
            self.util.error(f'unknown model type: \'{model_type}\'')
        self.model.load(run, epoch)

    def get_best_model(self):
        best_report = self.get_best_result(self.best_results)
        self.load_model(best_report)
        return self.model


    def get_best_result(self, reports):
        best_r = Reporter([], [], 0, 0)
        if self.util.exp_is_classification():
            measure = self.util.config_val('MODEL', 'measure', 'uar')
            best_r = self.search_best_result(reports, 'ascending')
        else:
            measure = self.util.config_val('MODEL', 'measure', 'mse')
            if measure == 'mse':
                best_r = self.search_best_result(reports, 'descending')
            elif measure == 'ccc':
                best_r = self.search_best_result(reports, 'ascending')
        return best_r

    def search_best_result(self, reports, order):
        best_r = Reporter([], [], 0, 0)
        if order == 'ascending':
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
        return best_r