# Nkululeko
A project to detect speaker characteristics with machine learning experiments with a very high level interface.

## Overview
The idea is to have a very high level framework (based on e.g. sklearn and pytorch) that can be used by people not being experienced programmers as they mainly have to adapt the initialization parameter files.
[Here's a blog post on how to set up nkululeko on your computer.](http://blog.syntheticspeech.de/2021/08/30/how-to-set-up-your-first-nkululeko-project/)

There is a central "experiment" class that can be used by own experiments, two examples are given with
* [exp_emodb.py](exp_emodb.py) ([configuration](exp_emodb.ini)), using SVM classifier
* [exp_emodb_mlp.py](exp_emodb_mlp.py) ([configuration](exp_emodb_mlp.ini)), using MLP classifier
  
An idea of the framework should give this UML sketch (not really valid any more, but to give you an idea).
![sketch](images/ml-experiment.jpg)

Currently the following classifiers are implemented (integrated from sklearn):
* SVM
* SVR
* XGB
* XGR
* MLP

Here's [a movie that shows the progress of classification done with nkululeko](https://youtu.be/6Y0M382GjvM)

## Usage
You could 
* use a generic main python file (like exp_emodb.py), 
* adapt the path to your nkululeko src 
* and then adapt an .ini file (again adapting at least the paths to src and data)
  
Here's [an overview on the ini-file options](./ini_file.md)