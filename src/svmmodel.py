# svmmodel.py

from sklearn import svm
from model import Model

class SVM_model(Model):
    """An SVM model"""
    def __init__(self, config, df_train, df_test, feats_train, feats_test):
        super().__init__(config, df_train, df_test, feats_train, feats_test)
        c = float(self.config['SVM']['C'])
        self.clf = svm.SVC(kernel='linear', C=c) # set up the classifier

    def set_C(self, c):
        """Set the C parameter"""
        self.clf.C = c

    def train(self):
        """Train the model"""
        target = self.config['DATA']['target']
        self.clf.fit(self.feats_train.df, self.df_train[target])

    def predict(self):
        """Predict the whole eval feature set"""
        return self.clf.predict(self.feats_test.df)