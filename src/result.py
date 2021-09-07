# result.py
import glob_conf

class Result:
    def __init__(self, test, train, loss):
        self.test = test
        self.train = train
        self.loss = loss 

    def to_string(self):
        return f'test: {self.test}, train: {self.train}, loss: {self.loss}'