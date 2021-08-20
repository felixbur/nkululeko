# Nkululeko
A project to implement a reusable framework for machine learning experiments.

The idea is to have a very high level framework (based on e.g. sklearn and pytorch) that can be used by people not being experienced programmers as they mainly have to adapt the initialization parameter files.

Currently the following classifiers are implemented:
* SVM
* XGB
* XGR
* MLP

There is a central "experiment" class that can be used by own experiments, two examples are given with
* exp_emodb.py, using XGB classifier
* exp_emodb_mlp.py, using MLP classifier
  
In the end i expect some kind of framework like in the UML sketch (not really valid any more, but to give you an idea).
![sketch](images/ml-experiment.jpg)
