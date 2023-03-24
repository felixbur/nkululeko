# model_gmm.py

from sklearn import mixture
from nkululeko.model import Model

class GMM_model(Model):
    """An GMM model"""

    is_classifier = True
    def __init__(self, df_train, df_test, feats_train, feats_test):
        super().__init__(df_train, df_test, feats_train, feats_test)
        n_components = int(self.util.config_val('MODEL', 'GMM_components', '4'))         
        covariance_type = self.util.config_val('MODEL', 'GMM_covariance_type', 'full') 
        self.clf = mixture.GaussianMixture(n_components=n_components, covariance_type=covariance_type)
        # set up the classifier

