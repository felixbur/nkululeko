# Nkululeko Pre-process for MSP-IMPROV dataset


Download the MSP-IMPROV dataset from https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html and place the extracted file in the `data` directory. You can `ln -sf` instead of hard copying the file.


```bash
# use the second version for more labels
$ python process_database2.py
$ cd ../..
$ python3 -m nkululeko.nkululeko --config data/msp-improv/exp_improv_audmodel_xgb.ini
...
# sample outputs
DEBUG dataset: loading train
DEBUG dataset: Loaded database train with 1622 samples: got targets: True, got speakers: False, got sexes: True
DEBUG dataset: train: loaded data with 1622 samples: got targets: True, got speakers: False, got sexes: True
DEBUG dataset: loading test
DEBUG dataset: Loaded database test with 5311 samples: got targets: True, got speakers: False, got sexes: True
DEBUG dataset: test: loaded data with 5311 samples: got targets: True, got speakers: False, got sexes: True
DEBUG dataset: loading dev
DEBUG dataset: Loaded database dev with 1505 samples: got targets: True, got speakers: False, got sexes: True
DEBUG dataset: dev: loaded data with 1505 samples: got targets: True, got speakers: False, got sexes: True
DEBUG experiment: loaded databases train,test,dev
DEBUG experiment: value for strategy not found, using default: traintest
DEBUG dataset: splitting database train with strategy train
DEBUG dataset: train: 0 samples in test and 1622 samples in train
DEBUG dataset: value for strategy not found, using default: train_test
DEBUG dataset: splitting database test with strategy test
DEBUG dataset: test: 5311 samples in test and 0 samples in train
DEBUG dataset: value for strategy not found, using default: train_test
DEBUG dataset: splitting database dev with strategy train
DEBUG dataset: dev: 0 samples in test and 1505 samples in train
DEBUG dataset: value for strategy not found, using default: train_test
DEBUG experiment: value for filter.sample_selection not found, using default: all
DEBUG experiment: value for type not found, using default: dummy
DEBUG experiment: Categories test: ['A' 'H' 'N' 'S']
DEBUG experiment: Categories train: ['A' 'H' 'N' 'S']
DEBUG nkululeko: train shape : (2880, 8), test shape:(4918, 8)
DEBUG featureset: value for aud.model not found, using default: ./audmodel/
DEBUG featureset: value for device not found, using default: cpu
DEBUG featureset: value for store_format not found, using default: pkl
DEBUG featureset: extracting audmodel embeddings, this might take a while...
DEBUG feature_extractor: audmodel: shape : (2880, 1024)                                             
DEBUG featureset: value for aud.model not found, using default: ./audmodel/
DEBUG featureset: value for device not found, using default: cpu
DEBUG featureset: value for store_format not found, using default: pkl
DEBUG featureset: extracting audmodel embeddings, this might take a while...
DEBUG feature_extractor: audmodel: shape : (4918, 1024)                                             
DEBUG experiment: All features: train shape : (2880, 1024), test shape:(4918, 1024)
DEBUG scaler: scaling features based on training set
DEBUG runmanager: run 0
/home/bagus/github/nkululeko/.env/lib/python3.8/site-packages/xgboost/sklearn.py:1395: UserWarning: `use_label_encoder` is deprecated in 1.7.0.
  warnings.warn("`use_label_encoder` is deprecated in 1.7.0.")
DEBUG modelrunner: run: 0 epoch: 0: result: test: 0.615 UAR
DEBUG modelrunner: plotting confusion matrix to train_test_dev_xgb_audmodel__0_000_cnf
DEBUG runmanager: value for measure not found, using default: uar
DEBUG reporter: labels: ['A' 'H' 'N' 'S']
DEBUG reporter: result per class (F1 score): [0.585, 0.707, 0.7, 0.454]
DEBUG experiment: Save experiment: Can't pickle local object: cannot pickle 'onnxruntime.capi.onnxruntime_pybind11_state.InferenceSession' object
DEBUG experiment: Done, used 18204.228 seconds
DONE
```