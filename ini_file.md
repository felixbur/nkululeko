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
* **save**: save the experiment as a pickle file to be restored again later (True or False)
  * save = False

### DATA
* **type**: just a flag now to mark continous data, so it can be binned to categorical data (using *bins* and *labels*)
  * type = continuous
* **databases**: list of databases to be used in the experiment
  * databases = ['emodb', 'timit']
* **strategy**: how the databases should be used, either *train_test* or *cross_data*
  * strategy = train_test
* **trains**: if *strategy* = cross_data, denote the train databases
* **tests**: if *strategy* = cross_data, denote the test databases
* **db_name**: path with audformatted repository for each database listed in 'databases*
  * emodb = /home/data/audformat/emodb/
* **db_name.type**: type of storage, e.g. audformat database or 'csv' (needs header: file,speaker,task)
  * emodb.type = audformat
* * **db_name.mapping**: mapping python dictionary to map between categories for cross-database experiments
  * emodb.mapping = {'anger':'angry', 'happiness':'happy', 'sadness':'sad', 'neutral':'neutral'}
* **db_name.split_strategy**: How to identify sets for train/development data splits within one database
  * emodb.split_strategy = reuse
  * Possible values:
    * **database**: default (*task*.train and *task*.test)
    * **specified**: specifiy the tables (an opportunity to assign multiple or no tables to train or dev set)
      * emodb.test_tables = ['emo.test', 'emo.train']
      * emodb.train_tables = ['emo.train']
    * **speaker_split**: split samples randomly but speaker disjunct, given a percentage of speakers for the test set.
      * emodb.testsplit = 50
    * **reuse**: reuse the splits after a *speaker_split* run to save time with feature extraction.
* **db_name.target_tables**: tables that containes the target / speaker / sex labels
  * emodb.target_tables = ['emotion']
* **db_name.files_tables**: tables that containes the audio file names
  * emodb.files_tables = ['files']
* **db_name.limit**: maximum number of samples per table (for testing with very large data mainly)
  * emodb.limit = 20
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
    * **os**: [open smile features](https://audeering.github.io/opensmile-python/)
      * set = eGeMAPSv02 *(features set)*
      * level = functionals *(or lld: feature level)*
    * **spectra**: Melspecs for convolutional networks
    * **trill**: [TRILL embeddings](https://ai.googleblog.com/2020/06/improving-speech-representations-and.html) from Google
    * **wav2vec**: [Wav2vec2 embeddings](https://huggingface.co/facebook/wav2vec2-large-robust-ft-swbd-300h) from facebook
      * **model** = = *path to the wav2vec2 model folder*
    * **mld**: [mid-level-descriptors](http://www.essv.de/paper.php?id=447)
      * mld = *path to the mld sources folder*
      * min_syls = *minimum number of syllables*
      * with_os = *with added opensmile eGemaps features*
      * features = *python list of selected features to be used (all others ignored)*
    * **xbow**: [open crossbow](https://github.com/openXBOW) features codebook computed from open smile features
      * size = 500 *(codebook size)*
      * assignments = 1 *(number of words in the bag representation where the counter is increased for each input LLD)*
      * with_os = *with added opensmile eGemaps features*
* **needs_feature_extraction**: if features should be extracted newly even if already stored
  * needs_feature_extraction = False
* **scale**: scale the features
  * scale=standard
  * possible values:
    * **standard**: z-transformation (mean of 0 and stdv of 1) based on training set
    * **speaker**: like *standard* but based on individual speaker sets (also for test)
* **set**: name of opensmile feature set, e.g. eGeMAPSv02, ComParE_2016, GeMAPSv01a, eGeMAPSv01a
  * set = eGeMAPSv02
* **level**: level of opensmile features
  * level = functional
  * possible values:
    * **functional**: aggregated over the whole utterance
    * **lld**: low level descriptor: framewise

### MODEL
* **type**: type of classifier
  * type = svm
  * possible values:
    * **svm**: Support Vector Machine 
      * C_val = 0.001
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
* **save**: whether to save all model states (per epoch) to disk (True or False)
  * save = False
* **loss_function**: A  loss function for ANN models
  * loss_function = mse
  * possible values:
    * **mse**: mean squared error
    * **1-ccc**: concordance correlation coefficient
* **measure**: A measure to report progress with regression experiments (classification is UAR)
  * measure = mse
  * possible values:
    * **mse**: mean squared error
    * **ccc**: concordance correlation coefficient
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
* **value_counts** plot statistics for each database and the train/dev splits (in the *image_dir*)
  * value_counts = 1
* **tsne** make a tsne plot to get a feeling how the features might perform
  * tsne = 1