# Overview on options for the nkululeko framework
to be specified in an .ini file, [config parser syntax](https://zetcode.com/python/configparser/)

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
  *  name = emodb_exp
*  **fig_dir**: (relative to *root*) folder for plots
  * fig_dir = ./images/
* **runs**: number of runs (e.g. to average over random initializations)
  * runs = 1
* **epochs**: number of epochs for ANN training
  * epochs = 50

### DATA
* **databases**: list of databases to be used in the experiment
  * databases = ['emodb', 'timit']
* **db_name**: path with audformatted repository for each database listed in 'databases*
  * emodb = /home/data/audformat/emodb/
* **db_name.mapping**: mapping python dictionary to map between categories for cross-database experiments
  * emodb.mapping = {'anger':'angry', 'happiness':'happy', 'sadness':'sad', 'neutral':'neutral'}
* **db_name.split_strategy**: How to identify sets for train/development data splits within one database
  * emodb.split_strategy = reuse
  * Possible values:
    * **database**: default (*task*.train and *task*.test)
    * **specified**: specifiy the tables (an opportunity to assign multiple or no tables to train or dev set)
      * emodb.test_tables = ['emo.test', 'emo.train']
    * **speaker_split**: split samples randomly but speaker disjunct, given a percentage of speakers for the test set.
      * emodb.testsplit = 30
    * **reuse**: reuse the splits after a *speaker_split* run to save time with feature extraction.
* **target**: the task name, e.g. *age* or *emotion*
  * target = emotion
* **labels**: for classification experiments: the names of the categories (is also used for regression when binning the values)
  * labels = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']

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