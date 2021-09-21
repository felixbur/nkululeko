import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
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
import math

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
                measure = self.util.config_val('MODEL', 'measure', 'mse')
                if measure == 'mse':
                    self.MEASURE = 'MSE'
                    self.result.test = mean_squared_error(self.truths, self.preds)
                elif measure == 'ccc':
                    self.MEASURE = 'CCC'
                    self.result.test = self.ccc(self.truths, self.preds)
                    if math.isnan(self.result.test):
                        self.util.debug(self.truths)
                        self.util.debug(self.preds)
                        self.util.error(f'result is NAN')
                else:
                    self.util.error(f'unknown measure: {measure}')

                # train and loss are being set by the model

    def set_id(self, run, epoch):
        """Make the report identifiable with run and epoch index"""
        self.run = run
        self.epoch = epoch

    def continuous_to_categorical(self):
        bins = ast.literal_eval(glob_conf.config['DATA']['bins'])
        self.truths = np.digitize(self.truths, bins)-1
        self.preds = np.digitize(self.preds, bins)-1

    def plot_confmatrix(self, plot_name, epoch): 
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
        if epoch != 0:
            plt.title(f'Confusion Matrix, UAR: {uar:.3f}, Epoch: {epoch}')
        else:
            plt.title(f'Confusion Matrix, UAR: {uar:.3f}')

        plt.savefig(fig_dir+plot_name)
        fig.clear()
        plt.close(fig)

    def print_results(self):
        res_dir = self.util.get_path('res_dir')
        if self.util.exp_is_classification():
            data_type = self.util.config_val('DATA', 'type', 'whatever')
            if data_type == 'continuous' or data_type == 'continous':
                labels = ast.literal_eval(glob_conf.config['DATA']['labels'])
            else:
                labels = glob_conf.label_encoder.classes_
            try:
                rpt = classification_report(self.truths, self.preds, target_names=labels)
            except ValueError:
                self.util.debug('Reporter: caught a ValueError when trying to get classification_report')
                rpt = self.result.to_string()
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
        filenames =  glob.glob(fig_dir+f'{self.util.get_plot_name()}*_?_???_cnf.png')
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        fps = self.util.config_val('PLOT', 'fps', '1')
        imageio.mimsave(fig_dir+out_name, images, fps=int(fps))

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
        if (self.util.exp_is_classification()):
            # scale up UAR
            results = results*100
            train_results = train_results*100
        plt.figure(dpi=200)
        plt.plot(train_results, 'green', label='train set') 
        plt.plot(results, 'red', label='dev set')
        plt.plot(losses, 'grey', label='losses')
        plt.xlabel('epochs')
        plt.ylabel(self.MEASURE)
        plt.legend()
        plt.savefig(fig_dir+ out_name)
        plt.close()        

    @staticmethod
    def ccc(ground_truth, prediction):
        mean_gt = np.mean(ground_truth, 0)
        mean_pred = np.mean(prediction, 0)
        var_gt = np.var (ground_truth, 0)
        var_pred = np.var (prediction, 0)
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        cor = sum (v_pred * v_gt) / (np.sqrt(sum(v_pred ** 2)) * np.sqrt(sum(v_gt ** 2)))
        sd_gt = np.std(ground_truth)
        sd_pred = np.std(prediction)
        numerator=2*cor*sd_gt*sd_pred
        denominator=var_gt+var_pred+(mean_gt-mean_pred)**2
        ccc = numerator/denominator
        return 1-ccc