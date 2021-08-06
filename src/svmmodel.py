# svmmodel.py

from sklearn import svm
from model import Model

class SVM_model(Model):
    """An SVM model"""
    clf = svm.SVC(kernel='linear', C=.001) # set up the classifier

    def set_C(self, c):
        """Set the C parameter"""
        self.clf.C = c

    def train(self):
        """Train the model"""
        self.clf.fit(self.feats_train.df, self.df_train['emotion'])

    def predict(self):
        """Predict the whole eval feature set"""
        return self.clf.predict(self.feats_test.df)