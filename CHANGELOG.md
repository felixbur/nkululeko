Changelog
=========

Version 0.26.0
-------------
* added the rename_speakers funcionality to prevent identical speaker names in datasets

Version 0.25.1
-------------
* fixed bug that no features were chosen if not selected

Version 0.25.0
-------------
* made selectable features universal for feature sets

Version 0.24.0
-------------
* added multiple feature sets (will simply be concatenated)

Version 0.23.0
-------------
* added selectable features for Praat interface

Version 0.22.0
-------------
* added David R. Feinberg's Praat features, praise also to parselmouth

Version 0.21.0
-------------

* Revoked 0.20.0
* Added support for only_test = True, to enable later testing of trained models with new test data

Version 0.20.0
-------------

* implemented reuse of trained and saved models

Version 0.19.0
-------------

* added "max_duration_of_sample" for datasets


Version 0.18.6
-------------

* added support for learning and dropout rate as argument


Version 0.18.5
-------------

* added support for epoch number as argument
  
Version 0.18.4
-------------

* added support for ANN layers as arguments

Version 0.18.3
-------------

* added reuse of test and train file sets
* added parameter to scale continous target values: target_divide_by


Version 0.18.2
-------------

* added preference of local dataset specs to global ones
  
Version 0.18.1
-------------

* added regression value display for confusion matrices

Version 0.18.0
-------------

* added leave one speaker group out

Version 0.17.2
-------------

* fixed scaler, added robust



Version 0.17.0
-------------

* Added minimum duration for test samples


Version 0.16.4
-------------

* Added possibility to combine predictions per speaker (with mean or mode function)

Version 0.16.3
-------------

* Added minimal sample length for databases


Version 0.16.2
-------------

* Added k-fold-cross-validation for linear classifiers

Version 0.16.1
-------------

* Added leave-one-speaker-out for linear classifiers


Version 0.16.0
-------------

* Added random sample splits

