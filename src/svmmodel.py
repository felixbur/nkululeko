# svmmodel.py

from sklearn import svm
from model import Model

class SVM_model(Model):
    clf = svm.SVC(kernel='linear', C=.001)

    def train(self):
        self.clf.fit(self.feats_train.df, self.df_train['emotion'])

    def predict(self):
        return self.clf.predict(self.feats_test.df)
        