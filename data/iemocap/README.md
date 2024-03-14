# README.md

IEMOCAP must be obtained from https://sail.usc.edu/iemocap/, unzip the file 
`IEMOCAP_full_release.zip` and placed the extracted file in the `data` directory. 
You can `ln -sf` instead of hard copying the file.

## Process database and run experiment
```bash
$ python process_database.py
$ cd ../..
$ python3 -m nkululeko.nkululeko --config data/iemocap/exp_iemocap_audmodel_xgb.ini
...
# sample outputs
DEBUG modelrunner: run: 0 epoch: 0: result: test: 0.716 UAR
DEBUG modelrunner: plotting confusion matrix to train_test_dev_xgb_audmodel__0_000_cnf
DEBUG runmanager: value for measure not found, using default: uar
DEBUG reporter: labels: ['ang' 'hap' 'neu' 'sad']
DEBUG reporter: result per class (F1 score): [0.759, 0.664, 0.685, 0.697]
DEBUG experiment: Save experiment: Can't pickle local object: cannot pickle 'onnxruntime.capi.onnxruntime_pybind11_state.InferenceSession' object
DEBUG experiment: Done, used 16361.050 seconds
```
