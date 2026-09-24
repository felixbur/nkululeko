# model_svm.py

from sklearn import svm

from nkululeko.models.model import Model


class SVM_model(Model):
    """An SVM model"""

    is_classifier = True

    def __init__(self, df_train, df_test, feats_train, feats_test):
        super().__init__(df_train, df_test, feats_train, feats_test)
        self.name = "svm"
        c = float(self.util.config_val("MODEL", "C_val", "1"))
        kernel = self.util.config_val("MODEL", "kernel", "rbf")
        # class_weight is deliberately not set here: Model.train() already
        # fits with a balanced sample_weight when [MODEL] class_weight =
        # True, and sklearn multiplies class_weight by sample_weight, so
        # setting both would square the per-class weighting (issue #448).
        self.clf = svm.SVC(
            kernel=kernel,
            C=c,
            gamma="scale",
            probability=True,
            random_state=42,  # for consistent result
        )  # set up the classifier

    def set_c(self, c):
        """Set the C parameter."""
        self.clf.C = c

    def get_type(self):
        return "svm"
