# result.py
import glob_conf

class Result:
    def __init__(self, test, train, loss):
        self.test = test
        self.train = train
        self.loss = loss 