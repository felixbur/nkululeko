# result.py


class Result:
    def __init__(self, test, train, loss, loss_eval, measure):
        self.test = test
        self.train = train
        self.loss = loss
        self.loss_eval = loss_eval
        self.measure = measure

    def get_result(self):
        return self.test

    def get_test_result(self):
        return f"test: {self.test:.3f} {self.measure}"

    def to_string(self):
        return (
            f"test: {self.test} {self.measure}, train:"
            f" {self.train} {self.measure}, loss: {self.loss}, eval-loss: {self.loss_eval}"
        )
