# runmanager.py

from svmmodel import SVM_model
from sklearn.metrics import recall_score

class Runmanager:
    config = None
    model = None 
    df_train, df_test, feats_train, feats_test = None, None, None, None



    def __init__(self, config, df_train, df_test, feats_train, feats_test):
        self.config = config
        self.df_train, self.df_test, self.feats_train, self.feats_test = df_train, df_test, feats_train, feats_test

    def do_runs(self):
        for r in range(int(self.config['RUN_MGR']['runs'])):
            self.model = SVM_model(self.config, self.df_train, self.df_test, self.feats_train, self.feats_test)
            for e in range(int(self.config['RUN_MGR']['epochs'])):
                self.model.train()
                results = self.model.predict()
                uar = self.evaluate(self.df_test['emotion'], results)
                print(f'run: {r} epoch: {e}: result: {uar}')


    def evaluate(self, truth, pred):
        return recall_score(truth, pred, average='macro')