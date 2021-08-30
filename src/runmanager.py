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
        for r in range(int(glob_conf.config['EXP']['runs'])):
            self.util.debug(f'run {r}')
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
            # for all epochs
            for e in range(int(glob_conf.config['EXP']['epochs'])):
                self.util.debug(f'epoch {e}')
                self.model.train()
                report = self.model.predict()
                plot_name = f'{self.util.get_exp_name()}_{r:02d}_{e:03d}_cnf.png'
                self.reports.append(report)                
                self.util.debug(f'run: {r} epoch: {e}: result: {self.reports[-1].get_result().test:.3f}')
                plot = self.util.config_val('PLOT', 'plot_epochs', 0)
                if plot:
                    report.plot_confmatrix(plot_name)
            try:
                # Is there a different name for a plot specified?
                plot_name = glob_conf.config['PLOT']['name']+'_cnf.png'
            except KeyError:
                pass
            # Do a final confusion matrix plot
            self.reports[-1].plot_confmatrix(plot_name)