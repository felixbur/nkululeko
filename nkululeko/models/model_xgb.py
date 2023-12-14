# xgbmodel.py

from xgboost import XGBClassifier
from nkululeko.models.model import Model


class XGB_model(Model):
    """An XGBoost model"""

    is_classifier = True

    clf = XGBClassifier()  # set up the classifier
