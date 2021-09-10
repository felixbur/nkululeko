# runmanager.py

from model_svm import SVM_model
from model_svr import SVR_model
from model_xgb import XGB_model
from model_xgr import XGR_model
from model_cnn import CNN_model
from model_mlp import MLP_model
from model_mlp_regression import MLP_Reg_model
from reporter import Reporter
from result import Result
import ast
from util import Util  
import glob_conf

class Runmanager:
    """Class to manage the runs of the experiment (e.g. when differs caused by random initialization)"""
    model = None  # The underlying model
    df_train, df_test, feats_train, feats_test = None, None, None, None # The dataframes


    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor setting up the dataframes"""
        self.df_train, self.df_test, self.feats_train, self.feats_test = df_train, df_test, feats_train, feats_test
        self.util = Util()
        self.target = glob_conf.config['DATA']['target']


    def do_runs(self):
        """Start the runs"""
        # for all runs
        for run in range(int(self.util.config_val('EXP', 'runs', 1))):
            self.util.debug(f'run {run}')
            # set the run index as global variable for reporting
            self.util.set_config_val('EXP', 'run', run)
            # initialze results
            self.reports = []
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
                self.model = CNN_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
            elif model_type=='mlp':
                self.model = MLP_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
            elif model_type=='mlp_reg':
                self.model = MLP_Reg_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
            else:
                self.util.error(f'unknown model type: \'{model_type}\'')
            plot_epochs = self.util.config_val('PLOT', 'plot_epochs', 0)
            # for all epochs
            for epoch in range(int(self.util.config_val('EXP', 'epochs', 1))):
                self.util.debug(f'epoch {epoch}')
                self.model.set_id(run, epoch)
                self.model.train()
                report = self.model.predict()
                report.set_id(run, epoch)
                plot_name = self.util.get_plot_name()+f'_{run}_{epoch:03d}_cnf.png'
                self.reports.append(report)                
                self.util.debug(f'run: {run} epoch: {epoch}: result: {self.reports[-1].get_result().test:.3f}')
                if plot_epochs:
                    self.util.debug(f'plotting conf matrix to {plot_name}')
                    report.plot_confmatrix(plot_name)
                store_models = self.util.config_val('MODEL', 'store', 0)
                if store_models:
                    self.model.store()
            if not plot_epochs:
                # Do a final confusion matrix plot
                self.util.debug(f'plotting final conf matrix to {plot_name}')
                self.reports[-1].plot_confmatrix(plot_name)
            # wrap up the run 
            plot_anim_progression = self.util.config_val('PLOT', 'plot_anim_progression', 0)
            if plot_anim_progression:
                plot_name_suggest = self.util.get_exp_name()
                plot_name = self.util.config_val('PLOT', 'name', plot_name_suggest)+'_conf_anim.gif'
                self.util.debug(f'plotting animated confusion to {plot_name}')
                self.reports[-1].make_conf_animation(plot_name)
            plot_epoch_progression = self.util.config_val('PLOT', 'plot_epoch_progression', 0)
            if plot_epoch_progression:
                plot_name_suggest = self.util.get_exp_name()
                plot_name = self.util.config_val('PLOT', 'name', plot_name_suggest)+'_epoch_progression.png'
                self.util.debug(f'plotting progression to {plot_name}')
                self.reports[-1].plot_epoch_progression(self.reports, plot_name)
            plot_best_model = self.util.config_val('PLOT', 'plot_best_model', 0)
            if plot_best_model:
                best_r = self.get_best_result()
                self.util.debug(f'best result with run {best_r.run} and epoch {best_r.epoch}: {best_r.result.test:.3f}')
                self.print_model(best_r)
            # finally, print out the numbers for this run
            self.reports[-1].print_results()


    def print_model(self, report):
        run = report.run
        epoch = report.epoch
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
            self.model = CNN_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
        elif model_type=='mlp':
            self.model = MLP_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
        elif model_type=='mlp_reg':
            self.model = MLP_Reg_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
        else:
            self.util.error(f'unknown model type: \'{model_type}\'')
        self.model.load(run, epoch)
        report = self.model.predict()
        plot_name_suggest = self.util.get_exp_name()
        plot_name = self.util.config_val('PLOT', 'name', plot_name_suggest)+f'_BEST_{run}_{epoch:03d}_cnf.png'
        self.util.debug(f'plotting conf matrix to {plot_name}')
        report.plot_confmatrix(plot_name)


    def get_best_result(self):
        best_r = Reporter([], [])
        if self.util.exp_is_classification():
            best_result = 0
            for r in self.reports:
                res = r.result.test
                if res > best_result:
                    best_result = res
                    best_r = r
        else:
            best_result = 10000
            for r in self.reports:
                res = r.result.test
                if res < best_result:
                    best_result = res
                    best_r = r
        return best_r
