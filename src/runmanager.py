# runmanager.py

from svmmodel import SVM_model
from sklearn.metrics import recall_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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
            self.model = SVM_model(self.config, self.df_train, self.df_test, self.feats_train, self.feats_test)
            # for all epochs
            for e in range(int(self.config['RUN_MGR']['epochs'])):
                self.model.train()
                results = self.model.predict()
                uar = self.evaluate(self.df_test['emotion'], results)#, True)
                print(f'run: {r} epoch: {e}: result: {uar}')


    def evaluate(self, truth, pred, plot=False):
        # Report evaluation data
        if plot:
            cm = confusion_matrix(truth, pred,  normalize = 'true')
            labels = ast.literal_eval(self.config['DATA']['labels'])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(cmap='gray')
            plt.savefig('filename.png')

        return recall_score(truth, pred, average='macro')
