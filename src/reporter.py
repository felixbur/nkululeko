import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score
from scipy.stats import pearsonr
import audplot
from util import Util 
import ast
import numpy as np

class Reporter:

    def __init__(self, config, truths, preds):
        self.config = config
        self.util = Util(config)
        self.truths = truths
        self.preds = preds

    def continuous_to_categorical(self):
        bins = ast.literal_eval(self.config['DATA']['bins'])
        self.truths = np.digitize(self.truths, bins)-1
        self.preds = np.digitize(self.preds, bins)-1

    def plot_confmatrix(self, plot_name): 
        fig_dir = self.util.get_path('fig_dir')
        sns.set()  # get prettier plots
        labels = ast.literal_eval(self.config['DATA']['labels'])
        plt.figure(figsize=[5, 5])
        plt.title('Confusion Matrix')
        audplot.confusion_matrix(self.truths, self.preds)
        # replace labels
        locs, _ = plt.xticks()
        plt.xticks(locs, labels)
        plt.yticks(locs, labels)
        plt.tight_layout()
        print(f'plotting conf matrix to {fig_dir+plot_name}')
        plt.savefig(fig_dir+plot_name)
        plt.close()

    def uar(self):
        return recall_score(self.truths, self.preds, average='macro')

    def pcc(self):
        return pearsonr(self.truths, self.preds)[0]
