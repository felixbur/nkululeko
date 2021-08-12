# runmanager.py

from svmmodel import SVM_model
from xgbmodel import XGB_model
from reporter import Reporter
import ast
from util import Util  

class Runmanager:
    """Class to manage the runs of the experiment (e.g. when differs caused by random initialization)"""
    config = None # The configuration
    model = None  # The underlying model
    df_train, df_test, feats_train, feats_test = None, None, None, None # The dataframes


    def __init__(self, config, df_train, df_test, feats_train, feats_test):
        """Constructor setting up the dataframes"""
        self.config = config
        self.df_train, self.df_test, self.feats_train, self.feats_test = df_train, df_test, feats_train, feats_test
        self.util = Util(config)
        self.results = []

    def do_runs(self):
        """Start the runs"""
        # for all runs
        for r in range(int(self.config['RUN_MGR']['runs'])):
            # intialize a new model
            model_type = self.config['MODEL']['type']
            if model_type=='svm':
                self.model = SVM_model(self.config, self.df_train, self.df_test, self.feats_train, self.feats_test)
            elif model_type=='xgb':
                self.model = XGB_model(self.config, self.df_train, self.df_test, self.feats_train, self.feats_test)
            # for all epochs
            for e in range(int(self.config['RUN_MGR']['epochs'])):
                self.model.train()
                results = self.model.predict()
                exp_name = self.config['EXP']['name']
                plot_name = f'{exp_name}_{str(r)}_{str(e)}_cnf.png'
                rpt = Reporter(self.config, self.df_test['emotion'], results)
                if self.util.exp_is_classification:
                    uar = rpt.uar()
                    self.results.append(uar)
                else: # regression
                    pcc = rpt.pcc()
                    self.results.append(pcc) 
                    
                print(f'run: {r} epoch: {e}: result: {self.results[-1]:.3f}')
            rpt.plot_confmatrix(plot_name)