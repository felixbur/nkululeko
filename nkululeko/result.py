# result.py

class Result:
    def __init__(self, test, train, loss, measure):
        self.test = test
        self.train = train
        self.loss = loss 
        self.measure = measure

    def get_test_result(self):
        return f'test: {self.test:.3f} {self.measure}'

    def to_string(self):
        return f'test: {self.test} {self.measure}, train: {self.train} {self.measure}, loss: {self.loss}'