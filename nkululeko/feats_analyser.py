# feats_analyser.py
import pandas as pd
from nkululeko.util import Util 
from nkululeko.plots import Plots
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

class FeatureAnalyser:
 

    def __init__(self, label, df_labels, df_features):
        self.util = Util()
        target = self.util.config_val('DATA', 'target', 'emotion')    
        self.y = df_labels[target]
        self.df_labels = df_labels
        self.X = df_features
        self.label = label


    def analyse(self):
        model_s = self.util.config_val('EXPL', 'model', 'log_reg')
        max_feat_num = int(self.util.config_val('EXPL', 'max_feats', '10'))
        importance = None
        self.util.debug('analysing features...')
        if self.util.exp_is_classification():
            if model_s == 'log_reg':
                model = LogisticRegression()
                model.fit(self.X, self.y)
                importance = model.coef_[0]
            elif model_s == 'tree':
                model = DecisionTreeClassifier()
                model.fit(self.X, self.y)
                importance = model.feature_importances_
                plot_tree = eval(self.util.config_val('EXPL', 'plot_tree', 'False'))
                if plot_tree:
                    plots = Plots()
                    plots.plot_tree(model, self.X)
            else:
                self.util.error(f'invalid analysis method: {model}')
        else: # regression experiment
            if model_s == 'lin_reg':
                model = LinearRegression()
                model.fit(self.X, self.y)
                importance = model.coef_
            elif model_s == 'tree':
                model = DecisionTreeRegressor()
                model.fit(self.X, self.y)
                importance = model.feature_importances_
            else:
                self.util.error(f'invalid analysis method: {model}')

        df_imp = pd.DataFrame({'feats':self.X.columns, 'importance':importance})
        df_imp = df_imp.sort_values(by='importance', ascending=False).iloc[:max_feat_num]
        ax = df_imp.plot(x='feats', y='importance', kind='bar')
        ax.set(title=f'{self.label} samples')
        plt.tight_layout()
        fig_dir = self.util.get_path('fig_dir')+'../' # one up because of the runs 
        exp_name = self.util.get_exp_name(only_data=True)
        format = self.util.config_val('PLOT', 'format', 'png')
        filename = f'{fig_dir}{exp_name}EXPL_{model_s}.{format}'
        plt.savefig(filename)
        fig = ax.figure
        fig.clear()
        plt.close(fig)
        # result file
        res_dir = self.util.get_path('res_dir')
        file_name = f'{res_dir}{self.util.get_exp_name(only_data=True)}EXPL_{model_s}.txt'
        with open(file_name, "w") as text_file:
            text_file.write(f'features in order of decreasing importance according to model {model_s}:\n'+
            f'{str(df_imp.feats.values)}\n')

        df_imp.to_csv(file_name, mode='a')

                # check if feature distributions should be plotted
        plot_feats = self.util.config_val('EXPL', 'feature_distributions', False)
        if plot_feats: 
            if self.util.exp_is_classification():
                for feature in df_imp.feats:
                    # plot_feature(self, title, feature, label, df_labels, df_features):
                    _plots = Plots()
                    _plots.plot_feature(plot_feats, feature, 'class_label', self.df_labels, self.X)
            else:
                self.util.debug('can\'t plot feature distributions if not classification')
