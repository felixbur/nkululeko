# result.py
from nkululeko.utils.util import Util


class Result:
    def __init__(self, test, train, loss, loss_eval, metric):
        self.test = test
        self.train = train
        self.loss = loss
        self.loss_eval = loss_eval
        self.metric = metric
        self.util = Util("Result")

    def get_result(self):
        return self.test

    def set_upper_lower(self, upper, lower):
        """Set the upper and lower bound of confidence interval."""
        self.upper = upper
        self.lower = lower

    def get_test_result(self):
        return f"test: {self.test:.3f} {self.metric}"

    def to_string(self):
        return (
            f"test: {self.test} {self.metric}, train:"
            f" {self.train} {self.metric}, loss: {self.loss}, eval-loss: {self.loss_eval}"
        )

    def test_result_str(self):
        result_s = self.util.to_3_digits_str(self.test)
        up_str = self.util.to_3_digits_str(self.upper)
        low_str = self.util.to_3_digits_str(self.lower)
        return f"{self.metric}: {result_s} ({up_str}/{low_str})"
