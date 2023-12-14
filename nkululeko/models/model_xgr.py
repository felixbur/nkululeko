# xgrmodel.py

from xgboost.sklearn import XGBRegressor
from nkululeko.models.model import Model


class XGR_model(Model):
    """An XGBoost model"""

    is_classifier = False

    clf = XGBRegressor()  # set up the regressor
