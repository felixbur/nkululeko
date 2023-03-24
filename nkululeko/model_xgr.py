# xgrmodel.py

from xgboost.sklearn import XGBRegressor
from nkululeko.model import Model

class XGR_model(Model):
    """An XGBoost model"""

    is_classifier = False

    clf = XGBRegressor(use_label_encoder=False) # set up the regressor

