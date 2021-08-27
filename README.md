# Nkululeko
A project to implement a reusable framework for machine learning experiments.

## Overview
The idea is to have a very high level framework (based on e.g. sklearn and pytorch) that can be used by people not being experienced programmers as they mainly have to adapt the initialization parameter files.

There is a central "experiment" class that can be used by own experiments, two examples are given with
* [exp_emodb.py](exp_emodb.py), using XGB classifier
* [exp_emodb_mlp.py](exp_emodb_mlp.py), using MLP classifier
  
An idea of the framework should give this UML sketch (not really valid any more, but to give you an idea).
![sketch](images/ml-experiment.jpg)

Currently the following classifiers are implemented:
* SVM
* XGB
* XGR
* MLP

Here's [a movie that shows the progress of classification done with nkululeko](https://youtu.be/6Y0M382GjvM)

## Usage
You could 
* use a generic main python file (like exp_emodb.py), 
* adapt the path to your nkululeko src 
* and then adapt an .ini file (again adapting at least the pathes to src and data)
  
Here's [an overview on the ini-file options](./ini_file.md)