Changelog
=========

Version 0.44.1
--------------
* bugfixing: feature importance: https://github.com/felixbur/nkululeko/issues/23
* bugfixing: loading csv database with filewise index https://github.com/felixbur/nkululeko/issues/24 


Version 0.45.0
--------------
* added auddim as features
* added FEATS store_format
* added device use to feat_audmodel

Version 0.44.1
--------------
* bugfixes

Version 0.44.0
--------------
* added scatter functions: tsne, pca, umap

Version 0.43.7
--------------
* added clap features

Version 0.43.6
--------------
* small bugs


Version 0.43.5
--------------
* because of difficulties with numba and audiomentations importing audiomentations only when augmenting

Version 0.43.4
--------------
* added error when experiment type and predictor don't match

Version 0.43.3
--------------
* fixed further bugs and added augmentation to the test runs

Version 0.43.2
--------------
* fixed a bug when running continuous variable as classification problem

Version 0.43.1
--------------
* fixed test_runs

Version 0.43.0
--------------
* added augmentation module based on audiomentation

Version 0.42.0
--------------
* age labels should now be detected in databases

Version 0.41.0
--------------
* added feature tree plot

Version 0.40.1
--------------
* fixed a bug: additional test database was not label encoded

Version 0.40.0
--------------
* added EXPL section and first functionality
* added test module (for test databases)

Version 0.39.0
--------------
* added feature distribution plots
* added  plot format

Version 0.38.3
--------------
* added demo mode with list argument

Version 0.38.2
--------------
* fixed a bug concerned with "no_reuse" evaluation

Version 0.38.1
--------------
* demo mode with file argument

Version 0.38.0
--------------
* fixed demo mode

Version 0.37.2
--------------
* mainly replaced pd.append with pd.concat


Version 0.37.1
--------------
* fixed bug preventing praat feature extraction to work

Version 0.37.0
--------------
* fixed bug cvs import not detecting multiindex 

Version 0.36.3
--------------
* published as a pypi module

Version 0.36.0
--------------
* added entry nkululeko.py script


Version 0.35.0
--------------
* fixed bug that prevented scaling (normalization)

Version 0.34.2
--------------
* smaller bug fixed concerning the loss_string

Version 0.34.1
--------------
* smaller bug fixes and tried Soft_f1 loss


Version 0.34.0
--------------
* smaller bug fixes and debug ouputs

Version 0.33.0
--------------
* added GMM as a model type

Version 0.32.0
--------------
* added audmodel embeddings as features

Version 0.31.0
--------------
* added models: tree and tree_reg
  
Version 0.30.0
--------------
* added models: bayes, knn and knn_reg

Version 0.29.2
--------------
* fixed hello world example


Version 0.29.1
--------------
* bug fix for 0.29


Version 0.29.0
--------------
* added a new FeatureExtractor class to import external data

Version 0.28.2
--------------
* removed some Pandas warnings
* added no_reuse function to database.load()

Version 0.28.1
--------------
* with database.value_counts show only the data that is actually used


Version 0.28.0
--------------
* made "label_data" configuration automatic and added "label_result"


Version 0.27.0
--------------
* added "label_data" configuration to label data with trained model (so now there can be train, dev and test set)

Version 0.26.1
--------------
* Fixed some bugs caused by the multitude of feature sets
* Added possibilty to distinguish between absolut or relative pathes in csv datasets

Version 0.26.0
--------------
* added the rename_speakers funcionality to prevent identical speaker names in datasets

Version 0.25.1
--------------
* fixed bug that no features were chosen if not selected

Version 0.25.0
--------------
* made selectable features universal for feature sets

Version 0.24.0
--------------
* added multiple feature sets (will simply be concatenated)

Version 0.23.0
--------------
* added selectable features for Praat interface

Version 0.22.0
--------------
* added David R. Feinberg's Praat features, praise also to parselmouth

Version 0.21.0
--------------

* Revoked 0.20.0
* Added support for only_test = True, to enable later testing of trained models with new test data

Version 0.20.0
--------------

* implemented reuse of trained and saved models

Version 0.19.0
--------------

* added "max_duration_of_sample" for datasets


Version 0.18.6
--------------

* added support for learning and dropout rate as argument


Version 0.18.5
--------------

* added support for epoch number as argument
  
Version 0.18.4
--------------

* added support for ANN layers as arguments

Version 0.18.3
--------------

* added reuse of test and train file sets
* added parameter to scale continous target values: target_divide_by


Version 0.18.2
--------------

* added preference of local dataset specs to global ones
  
Version 0.18.1
--------------

* added regression value display for confusion matrices

Version 0.18.0
--------------

* added leave one speaker group out

Version 0.17.2
--------------

* fixed scaler, added robust



Version 0.17.0
--------------

* Added minimum duration for test samples


Version 0.16.4
--------------

* Added possibility to combine predictions per speaker (with mean or mode function)

Version 0.16.3
--------------

* Added minimal sample length for databases


Version 0.16.2
--------------

* Added k-fold-cross-validation for linear classifiers

Version 0.16.1
--------------

* Added leave-one-speaker-out for linear classifiers


Version 0.16.0
--------------

* Added random sample splits

