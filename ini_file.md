# Overview on options for the nkululeko framework
* To be specified in an .ini file, [config parser syntax](https://zetcode.com/python/configparser/)
* Kind of all (well, most) values have defaults 

## Sections
### EXP

* **root**: experiment root folder 
  * root = /xxx/projects/nkululeko/
* **type**: the kind of experiment
  * type = classification
  * possible values:
    * **classification**: supervised learning experiment with restricted set of categories (e.g. emotion categories).
    * **regression**: supervised learning experiment with continous values (e.g. speaker age in years).
* **store**: (relative to *root*) folder for caches
  * store = ./store/
* **name**: a name for debugging output
  * name = emodb_exp
* **fig_dir**: (relative to *root*) folder for plots
  * fig_dir = ./images/
* **res_dir**: (relative to *root*) folder for result output
  * res_dir = ./results/
* **models_dir**: (relative to *root*) folder to save models
  * models_dir = ./models/
* **runs**: number of runs (e.g. to average over random initializations)
  * runs = 1
* **epochs**: number of epochs for ANN training
  * epochs = 1

### DATA
* **data_type**: just a flag now to mark continous data, so it can be binned to categorical data (using *bins* and *labels*)
  * data_type = continous
* **databases**: list of databases to be used in the experiment
  * databases = ['emodb', 'timit']
* **db_name**: path with audformatted repository for each database listed in 'databases*
  * emodb = /home/data/audformat/emodb/
* **db_name.type**: type of storage, e.g. audformat database or CSV (needs header: file,speaker,task)
  * emodb.type = audformat
* * **db_name.mapping**: mapping python dictionary to map between categories for cross-database experiments
  * emodb.mapping = {'anger':'angry', 'happiness':'happy', 'sadness':'sad', 'neutral':'neutral'}
* **db_name.split_strategy**: How to identify sets for train/development data splits within one database
  * emodb.split_strategy = reuse
  * Possible values:
    * **database**: default (*task*.train and *task*.test)
    * **specified**: specifiy the tables (an opportunity to assign multiple or no tables to train or dev set)
      * emodb.test_tables = ['emo.test', 'emo.train']
    * **speaker_split**: split samples randomly but speaker disjunct, given a percentage of speakers for the test set.
      * emodb.testsplit = 50
    * **reuse**: reuse the splits after a *speaker_split* run to save time with feature extraction.
* **target**: the task name, e.g. *age* or *emotion*
  * target = emotion
* **labels**: for classification experiments: the names of the categories (is also used for regression when binning the values)
  * labels = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
* **bins**: array of integers to be used for binning continous data 
  * bins  = [-100, 40, 50, 60, 70, 100]
### FEATS
* **type**: the type of features
  * type = os
  * possible values:
    * **os**: open smile features
    * **spectra**: Melspecs for convolutional networks
    * **mld**: mid-level-descriptors
      * min_syls = *minimum number of syllables*
      * with_os = *adding opensmile eGemaps features*
      * features = *python list of selected features to be used (all others ignored)*
* **scale**: scale the features
  * scale=standard
  * possible values:
    * **standard**: z-transformation (mean of 0 and stdv of 1) based on training set
    * **speaker**: like *standard* but based on individual speaker sets (also for test)

### MODEL
* **type**: type of classifier
  * type = svm
  * possible values:
    * **svm**: Support Vector Machine 
      * C = 0.001
    * **xgb**:XG-Boost
    * **svr**: Support Vector Regression
    * **xgr**: XG-Boost Regression
    * **mlp**: Multi-Layer-Perceptron for classification
    * **mlp_reg**: Multi-Layer-Perceptron for regression
    * **cnn**: Convolutional neural network (tbd)
* **tuning_params**: possible tuning parameters for x-fold optimization (for SVM, SVR, XGB and XGR)
  * tuning_params = ['subsample', 'n_estimators', 'max_depth']
    * subsample = [.5, .7]
    * n_estimators = [50, 80, 200]
    * max_depth = [1, 6]
* **scoring**: scoring measure for the optimization
  * scoring = recall_macro
* **layers**: layer outline (number of hidden layers and number of neurons per layer) for the MLP as a python dictionary
  * layers = {'l1':8, 'l2':4}
* **class_weight**: add class_weight to linear classifier (XGB, SVM) fit methods for imbalanced data
  * class_weight = 1
* **store**: whether to save all model states (per epoch) to disk
  * store = 1
* **loss_function**: A  loss function for ANN models
  * loss_function = mse
* **measure**: A measure to report progress (and decide for the best outcome)
  * measure = mse
* **learning_rate**: The learning rate for ANN models
  * learning_rate = 0.0001

### PLOT
* **name**: special name as a prefix for all plots (stored in *img_dir*).
  * name = my_special_config_within_the_experiment
* **plot_epochs**: whether to make a plot each for every epoch result.
  * plot_epochs = 1
* **plot_anim_progression**: generate an animated gif from the epoch plots
  * plot_anim_progression = 1
* **fps**: frames per second for the animated gif
  * fps = 1
* **plot_epoch_progression**: plot the progression of test, train and loss results over epochs
  * plot_epoch_progression = 1
* **plot_best_model**: search for the best performing model and plot conf matrix (needs *MODEL.store* to be turned on)
  * plot_best_model = 1
