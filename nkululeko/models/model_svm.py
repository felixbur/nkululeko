# model_svm.py

from sklearn import svm

from nkululeko.models.model import Model


class SVM_model(Model):
    """An SVM model"""

    is_classifier = True

    def __init__(self, df_train, df_test, feats_train, feats_test):
        super().__init__(df_train, df_test, feats_train, feats_test)
        self.name = "svm"
        c = float(self.util.config_val("MODEL", "C_val", "0.001"))
        if eval(self.util.config_val("MODEL", "class_weight", "False")):
            class_weight = "balanced"
        else:
            class_weight = None
        kernel = self.util.config_val("MODEL", "kernel", "rbf")
        self.clf = svm.SVC(
            kernel=kernel,
            C=c,
            gamma="scale",
            probability=True,
            class_weight=class_weight,
            random_state=42,  # for consistent result
        )  # set up the classifier

    def set_c(self, c):
        """Set the C parameter."""
        self.clf.C = c

    def get_type(self):
        return "svm"
