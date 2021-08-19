# runmanager.py

from model_svm import SVM_model
from model_xgb import XGB_model
from model_xgr import XGR_model
from model_cnn import CNN_model
from reporter import Reporter
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
        self.results = []
        self.target =  glob_conf.config['DATA']['target']


    def do_runs(self):
        """Start the runs"""
        # for all runs
        for r in range(int(glob_conf.config['RUN_MGR']['runs'])):
            self.util.debug(f'run {r}')
            # intialize a new model
            model_type = glob_conf.config['MODEL']['type']
            if model_type=='svm':
                self.model = SVM_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
            elif model_type=='xgb':
                self.model = XGB_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
            elif model_type=='xgr':
                self.model = XGR_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
            elif model_type=='cnn':
                self.model = CNN_model(self.df_train, self.df_test, self.feats_train, self.feats_test)
            # for all epochs
            for e in range(int(glob_conf.config['RUN_MGR']['epochs'])):
                self.util.debug(f'epoch {e}')
                self.model.train()
                results = self.model.predict()
                exp_name = glob_conf.config['EXP']['name']
                plot_name = f'{exp_name}_{str(r)}_{str(e)}_cnf.png'
                rpt = Reporter(self.df_test[self.target], results)
                if self.util.exp_is_classification():
                    uar = rpt.uar()
                    self.results.append(uar)
                else: # regression
                    pcc = rpt.pcc()
                    self.results.append(pcc) 
                    rpt.continuous_to_categorical()
                    
                print(f'run: {r} epoch: {e}: result: {self.results[-1]:.3f}')
            # see if there is a special plotname
            try:
                plot_name = glob_conf.config['PLOT']['name']+'_cnf.png'
            except KeyError:
                pass
            rpt.plot_confmatrix(plot_name)