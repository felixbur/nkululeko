# xgbmodel.py

from xgboost import XGBClassifier
from model import Model
import glob_conf
from reporter import Reporter
from result import Result
from sklearn.model_selection import GridSearchCV
import ast

class XGB_model(Model):
    """An XGBoost model"""
    clf = XGBClassifier() # set up the classifier

