# model_knn_reg.py

from sklearn.neighbors import KNeighborsRegressor

from nkululeko.models.model import Model


class KNN_reg_model(Model):
    """An KNN model"""

    is_classifier = False

    def __init__(self, df_train, df_test, feats_train, feats_test):
        super().__init__(df_train, df_test, feats_train, feats_test)
        self.name = "knn_reg"
        method = self.util.config_val("MODEL", "KNN_weights", "uniform")
        k = int(self.util.config_val("MODEL", "K_val", "5"))
        self.clf = KNeighborsRegressor(
            n_neighbors=k, weights=method
        )  # set up the classifier
