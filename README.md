# Nkululeko

## Description
A project to detect speaker characteristics by machine learning experiments with a high level interface based on [audformat](https://github.com/audeering/audformat).
The idea is to have a very high level framework (based on e.g. sklearn and pytorch) that can be used by people not being experienced programmers as they mainly have to adapt the initialization parameter files.
[Here's a blog post on how to set up nkululeko on your computer.](http://blog.syntheticspeech.de/2021/08/30/how-to-set-up-your-first-nkululeko-project/)


## Installation
Download the sources via git clone or zip export to your hard drive.
Include the classes via 
```
import sys
sys.path.append("./src")
```
in your main python file and use them.
An example is the [my_experiment.py](my_experiment.py) file.

## Usage
There is a central "experiment" class that can be used by own experiments, two examples are given with
* [my_experiment.py](demos/my_experiment.py) ([configuration](demos/exp_emodb.ini)), using SVM classifier
* [configuration](demos/exp_emodb_mlp.ini), using MLP classifier

Here are some other use case demonstrations:
* [On-th-fly classification with the best model](demos/demo_best_model.py)
* [Re-generate the progressing confusion matrix animation wit a different framerate](demos/plot_faster_anim.py)


The framework is targeted at the speech domain and supports experiments where different classifiers are combined with different feature extractors.

Here's a rough UML-like sketch of the framework.
![sketch](images/ml-experiment.jpg)

Currently the following linear classifiers are implemented (integrated from sklearn):
* SVM, SVR, XGB, XGR
  and the following ANNs
* MLP, CNN (tbd)

Here's [an animation that shows the progress of classification done with nkululeko](https://youtu.be/6Y0M382GjvM)

### Initialization file
You could 
* use a generic main python file (like my_experiment.py), 
* adapt the path to your nkululeko src 
* and then adapt an .ini file (again fitting at least the paths to src and data)
  
Here's [an overview on the ini-file options](./ini_file.md)

### Features
* Classifiers: XGB, XGR, SVM, SVR, MLP
* Feature extractors: opensmile, TRILL embeddings (experimental)
* Feature scaling
* Label encoding
* Binning (continuous to categorical)

### Outlook
* Classifiers: CNN
* Feature extractors: mid level descriptors, Mel-spectra, embeddings
* Online demo interface for trained models 

## Licence
Nkululeko can be used under the [MIT license](https://choosealicense.com/licenses/mit/)