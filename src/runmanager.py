# runmanager.py

from svmmodel import SVM_model
from xgbmodel import XGB_model
from sklearn.metrics import recall_score
# from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import audplot
import matplotlib.pyplot as plt
import seaborn as sns
import ast

class Runmanager:
    """Class to manage the runs of the experiment (e.g. when differs caused by random initialization)"""
    config = None # The configuration
    model = None  # The underlying model
    df_train, df_test, feats_train, feats_test = None, None, None, None # The dataframes



    def __init__(self, config, df_train, df_test, feats_train, feats_test):
        """Constructor setting up the dataframes"""
        self.config = config
        self.df_train, self.df_test, self.feats_train, self.feats_test = df_train, df_test, feats_train, feats_test

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
                uar = self.evaluate(self.df_test['emotion'], results, plot_name)
                print(f'run: {r} epoch: {e}: result: {uar:.3f}')


    def evaluate(self, truth, pred, plot_name = ''):
        # Report evaluation data
        if plot_name:
            fig_dir = self.config['EXP']['fig_dir']
            sns.set()  # get prettier plots
            fig = plt.figure()
            cm = confusion_matrix(truth, pred,  normalize = 'true')
            labels = ast.literal_eval(self.config['DATA']['labels'])
            plt.figure(figsize=[2.8, 2.5])
            plt.title('Confusion Matrix')
            audplot.confusion_matrix(truth, pred)

            # replace labels
            locs, _ = plt.xticks()
            plt.xticks(locs, labels)
            plt.yticks(locs, labels)

            plt.tight_layout()
            plt.savefig(fig_dir+plot_name)

        return recall_score(truth, pred, average='macro')
