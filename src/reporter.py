import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score
from scipy.stats import pearsonr
import audplot
from util import Util 
import ast
import numpy as np
import glob_conf
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

class Reporter:

    def __init__(self, truths, preds):
        self.util = Util()
        self.truths = truths
        self.preds = preds

    def continuous_to_categorical(self):
        bins = ast.literal_eval(glob_conf.config['DATA']['bins'])
        self.truths = np.digitize(self.truths, bins)-1
        self.preds = np.digitize(self.preds, bins)-1

    def plot_confmatrix_mew(self, plot_name): 
        fig_dir = self.util.get_path('fig_dir')
        labels = ast.literal_eval(glob_conf.config['DATA']['labels'])
        plt.figure()  # figsize=[5, 5]
        cm = confusion_matrix(self.truths, self.preds,  normalize = 'true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(cmap='Blues')
        print(f'plotting conf matrix to {fig_dir+plot_name}')
        plt.title('Confusion Matrix')
        plt.savefig(fig_dir+plot_name)
        plt.close()


    def plot_confmatrix(self, plot_name): 
        fig_dir = self.util.get_path('fig_dir')
        sns.set()  # get prettier plots
        labels = ast.literal_eval(glob_conf.config['DATA']['labels'])
        plt.figure()  # figsize=[5, 5]
        plt.title('Confusion Matrix')
        plt.ylabel('UAR')
        audplot.confusion_matrix(self.truths, self.preds)
        # replace labels
        locs, _ = plt.xticks()
        plt.xticks(locs, labels)
        plt.yticks(locs, labels)
        plt.tight_layout()
        print(f'plotting conf matrix to {fig_dir+plot_name}')
        plt.savefig(fig_dir+plot_name)
        plt.close()
        print('truths')
        print(self.truths.values)
        print('preds')
        print(self.preds)
        print('labels')
        print(labels)

    def uar(self):
        return recall_score(self.truths, self.preds, average='macro')

    def pcc(self):
        return pearsonr(self.truths, self.preds)[0]
