class Experiment:
    name = ''
    datasets = []
    def __init__(self, name):
        self.name = name
        self.datasets = []

    def add_dataset(self, ds):
        self.datasets.append(ds)

    

