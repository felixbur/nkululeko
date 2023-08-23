# Nkululeko dataset processing for MSP-Podcast dataset

MSP-Podcast can be obtained from [here]https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html).
We used version 1.8.0 of the dataset.

```bash
$ python3 process_database.py
$ cd ../..
$ python3 -m nkululeko.nkululeko --config data/msp-podcast/exp_podcast_audmodel_xgb.ini
# sample output
...
DEBUG modelrunner: run: 0 epoch: 0: result: test: 0.399 UAR
DEBUG modelrunner: plotting confusion matrix to train_test_dev_xgb_audmodel__0_000_cnf
DEBUG runmanager: value for measure not found, using default: uar
DEBUG reporter: labels: ['ang' 'hap' 'neu' 'sad']
DEBUG reporter: result per class (F1 score): [0.275, 0.491, 0.788, 0.118]
DEBUG experiment: Save experiment: Can't pickle local object: cannot pickle 'onnxruntime.capi.onnxruntime_pybind11_state.InferenceSession' object
DEBUG experiment: Done, used 129520.888 seconds
```