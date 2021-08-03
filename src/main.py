import sys

import experiment as exp
import dataset as ds

def main():
    expr = exp.Experiment('my cool experiment')
    print(expr.name)
    data = ds.Dataset('emodb')
    expr.add_dataset(data)
    for d in expr.datasets:
        print(d.name)
if __name__ == "__main__":
    main()
