import matplotlib.pyplot as plt
import seaborn as sns
from util import Util 
import ast
import numpy as np
import glob_conf
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from scipy.stats import pearsonr
from result import Result

class Reporter:

    def __init__(self, truths, preds):
        self.util = Util()
        self.truths = truths
        self.preds = preds
        self.result = Result(0, 0, 0)
        if self.util.exp_is_classification():
            self.result.test = recall_score(self.truths, self.preds, average='macro')
            self.result.loss = 1 - accuracy_score(self.truths, self.preds)
        else:
            # regression experiment
            self.result.test = mean_squared_error(self.truths, self.preds)
            # train and loss are being set by the model

    def continuous_to_categorical(self):
        bins = ast.literal_eval(glob_conf.config['DATA']['bins'])
        self.truths = np.digitize(self.truths, bins)-1
        self.preds = np.digitize(self.preds, bins)-1

    def plot_confmatrix(self, plot_name): 
        if not self.util.exp_is_classification():
            self.continuous_to_categorical()

        fig_dir = self.util.get_path('fig_dir')
        try:
            labels = glob_conf.label_encoder.classes_
        except AttributeError:
            labels = ast.literal_eval(glob_conf.config['DATA']['labels'])

        fig = plt.figure()  # figsize=[5, 5]
        uar = recall_score(self.truths, self.preds, average='macro')
        cm = confusion_matrix(self.truths, self.preds,  normalize = None) #normalize must be one of {'true', 'pred', 'all', None}
        try:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(cmap='Blues')
        except ValueError:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels).remove('neutral')).plot(cmap='Blues')

        print(f'plotting conf matrix to {fig_dir+plot_name}')
        plt.title(f'Confusion Matrix, UAR: {uar:.3f}')
        plt.savefig(fig_dir+plot_name)
        fig.clear()
        plt.close(fig)

    def print_results(self):
        res_dir = self.util.get_path('res_dir')
        if self.util.exp_is_classification():
            labels = glob_conf.label_encoder.classes_
            rpt = classification_report(self.truths, self.preds, target_names=labels)
            file_name = f'{res_dir}{self.util.get_exp_name()}.txt'
            with open(file_name, "w") as text_file:
                text_file.write(rpt)
        else: # regression
            mse = self.result.test
            r2 = r2_score(self.truths, self.preds)
            pcc = pearsonr(self.truths, self.preds)[0]
            file_name = f'{res_dir}{self.util.get_exp_name()}.txt'
            with open(file_name, "w") as text_file:
                text_file.write(f'mse: {mse:.3f}, r_2: {r2:.3f}, pcc {pcc:.3f}')


    def get_result(self):
        return self.result