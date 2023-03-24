# model_tree_reg.py

from sklearn.tree import DecisionTreeRegressor
from nkululeko.model import Model

class Tree_reg_model(Model):

    is_classifier = False

    """An Tree model"""
    def __init__(self, df_train, df_test, feats_train, feats_test):
        super().__init__(df_train, df_test, feats_train, feats_test)
        self.clf = DecisionTreeRegressor() # set up the classifier

