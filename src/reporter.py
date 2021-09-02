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
import imageio
import glob

class Reporter:

    def __init__(self, truths, preds):
        """Initialization with ground truth und predictions vector"""
        self.util = Util()
        self.truths = truths
        self.preds = preds
        self.result = Result(0, 0, 0)
        self.run = 0
        self.epoch = 0
        if len(truths)>0 and len(preds)>0:
            if self.util.exp_is_classification():
                self.MEASURE = 'UAR'
                self.result.test = recall_score(self.truths, self.preds, average='macro')
                self.result.loss = 1 - accuracy_score(self.truths, self.preds)
            else:
                # regression experiment
                self.MEASURE = 'MSE'
                self.result.test = mean_squared_error(self.truths, self.preds)
                # train and loss are being set by the model

    def set_id(self, run, epoch):
        """Make the report identifiable with run and epoch index"""
        self.run = run
        self.epoch = epoch

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
        plt.title(f'Confusion Matrix, UAR: {uar:.3f}')
        plt.savefig(fig_dir+plot_name)
        fig.clear()
        plt.close(fig)

    def print_results(self):
        res_dir = self.util.get_path('res_dir')
        if self.util.exp_is_classification():
            if self.util.config_val('DATA', 'data_type', 'whatever') == 'continous':
                labels = ast.literal_eval(glob_conf.config['DATA']['labels'])
            else:
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

    def make_conf_animation(self, out_name):
        fig_dir = self.util.get_path('fig_dir')
        filenames =  glob.glob(fig_dir+'*_cnf.png')
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(fig_dir+out_name, images)

    def get_result(self):
        return self.result


    def plot_epoch_progression(self, reports, out_name):
        fig_dir = self.util.get_path('fig_dir')
        results, losses, train_results = [], [], []
        for r in reports:
            results.append(r.get_result().test)
            losses.append(r.get_result().loss)
            train_results.append(r.get_result().train)

        # do a plot per run
        # scale the losses so they fit on the picture
        losses, results, train_results = np.asarray(losses), np.asarray(results), np.asarray(train_results)
        losses = losses/2
        if (self.util.exp_is_classification()):
            # scale up UAR
            results = results*100
            train_results = train_results*100
        plt.figure(dpi=200)
        plt.plot(train_results, 'green', label='train set') 
        plt.plot(results, 'red', label='dev set')
        plt.plot(losses, 'grey', label='losses/2')
        plt.xlabel('epochs')
        plt.ylabel(self.MEASURE)
        plt.legend()
        plt.savefig(fig_dir+ out_name)
        plt.close()        