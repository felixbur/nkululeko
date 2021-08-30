# xgrmodel.py

from xgboost.sklearn import XGBRegressor
from model import Model
import glob_conf
from reporter import Reporter
from sklearn.model_selection import GridSearchCV
import ast


class XGR_model(Model):
    """An XGBoost model"""
    clf = XGBRegressor() # set up the regressor

