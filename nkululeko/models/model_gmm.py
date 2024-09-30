# model_gmm.py

import pandas as pd
from sklearn import mixture

from nkululeko.models.model import Model


class GMM_model(Model):
    """An GMM model"""

    is_classifier = True

    def __init__(self, df_train, df_test, feats_train, feats_test):
        super().__init__(df_train, df_test, feats_train, feats_test)
        self.name = "gmm"
        self.n_components = int(self.util.config_val("MODEL", "GMM_components", "4"))
        covariance_type = self.util.config_val("MODEL", "GMM_covariance_type", "full")
        self.clf = mixture.GaussianMixture(
            n_components=self.n_components,
            covariance_type=covariance_type,
            random_state=42,
        )
        # set up the classifier

    def get_predictions(self):
        """Use the predict_proba method of the GaussianMixture model to get
        probabilities. Create a DataFrame with these probabilities and return
        it along with the predictions."""
        probs = self.clf.predict_proba(self.feats_test)
        preds = self.clf.predict(self.feats_test)

        # Convert predictions to a list
        preds = preds.tolist()

        # Create a DataFrame for probabilities
        proba_df = pd.DataFrame(
            probs, index=self.feats_test.index, columns=range(self.n_components)
        )

        return preds, proba_df
