# model_tree.py

from sklearn.tree import DecisionTreeClassifier
from nkululeko.model import Model

class Tree_model(Model):
    """An SVM model"""
    def __init__(self, df_train, df_test, feats_train, feats_test):
        super().__init__(df_train, df_test, feats_train, feats_test)
        self.clf = DecisionTreeClassifier() # set up the classifier

