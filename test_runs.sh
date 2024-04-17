#!/bin/bash

# test explore module
function Explore {
    python -m nkululeko.explore --config tests/exp_emodb_explore_data.ini
    python -m nkululeko.explore --config tests/exp_emodb_explore_featimportance.ini
    python -m nkululeko.explore --config tests/exp_emodb_explore_scatter.ini
    python -m nkululeko.explore --config tests/exp_emodb_explore_features.ini
    python -m nkululeko.explore --config tests/exp_agedb_explore_data.ini
}
# test basic nkululeko
function Nkulu {
    python -m nkululeko.nkululeko --config tests/exp_emodb_os_xgb.ini 
    python -m nkululeko.nkululeko --config tests/exp_emodb_os_svm.ini
    python -m nkululeko.nkululeko --config tests/exp_emodb_os_knn.ini
    python -m nkululeko.nkululeko --config tests/exp_emodb_os_praat_xgb.ini
    python -m nkululeko.nkululeko --config tests/exp_emodb_featimport_xgb.ini
    python -m nkululeko.nkululeko --config tests/exp_emodb_cnn.ini
    python -m nkululeko.nkululeko --config tests/exp_emodb_balancing.ini
    python -m nkululeko.nkululeko --config tests/exp_emodb_audmodel_xgb.ini
    python -m nkululeko.nkululeko --config tests/exp_emodb_split.ini
    python -m nkululeko.nkululeko --config tests/exp_emodb_os_mlp.ini
    python -m nkululeko.nkululeko --config tests/exp_ravdess_os_xgb.ini
    python -m nkululeko.nkululeko --config tests/exp_agedb_os_xgr.ini 
    python -m nkululeko.nkululeko --config tests/exp_agedb_os_mlp.ini 
    python -m nkululeko.nkululeko --config tests/exp_agedb_class_os_xgb.ini 
}
# test augmentation
function Aug {
    python -m nkululeko.augment --config tests/exp_emodb_augment_os_xgb.ini
    python -m nkululeko.nkululeko --config tests/exp_emodb-aug_os_xgb.ini
    python -m nkululeko.augment --config tests/exp_emodb_random_splice_os_xgb.ini
    python -m nkululeko.nkululeko --config tests/exp_emodb_rs_os_xgb.ini
    python -m nkululeko.aug_train --config tests/emodb_aug_train.ini
}
# test prediction
function Pred {
    python -m nkululeko.predict --config tests/exp_emodb_predict.ini
    python -m nkululeko.nkululeko --config tests/emodb_demo.ini
}
# test demo
function Demo {
    python -m nkululeko.demo --config tests/exp_emodb_os_xgb.ini --list data/test/samples.csv
    python -m nkululeko.demo --config tests/exp_emodb_os_svm.ini --list data/test/samples.csv
    python -m nkululeko.demo --config tests/exp_emodb_os_knn.ini --list data/test/samples.csv
}
# test test module
function Test {
    python -m nkululeko.nkululeko --config tests/exp_emodb_os_xgb_test.ini
    python -m nkululeko.test --config tests/exp_emodb_os_xgb_test.ini
}
# test multidb
function Multi {
    python -m nkululeko.multidb --config tests/exp_multidb.ini
}
# test spotlight
function Spot {
    python -m nkululeko.explore --config tests/exp_explore.ini
}

for arg in "$@"; do
  if [[ "$arg" = --Explore ]] || [[ "$arg" = --all ]]; then
    Explore
  fi
  if [[ "$arg" = --Nkulu ]] || [[ "$arg" = --all ]]; then
    Nkulu
  fi
  if [[ "$arg" = --Aug ]] || [[ "$arg" = --all ]]; then
    Aug
  fi
  if [[ "$arg" = --Pred ]] || [[ "$arg" = --all ]]; then
    Pred
  fi
  if [[ "$arg" = --Demo ]] || [[ "$arg" = --all ]]; then
    Demo
  fi
  if [[ "$arg" = --Test ]] || [[ "$arg" = --all ]]; then
    Test
  fi
  if [[ "$arg" = --Multi ]] || [[ "$arg" = --all ]]; then
    Multi
  fi
  if [[ "$arg" = --Spot ]] || [[ "$arg" = --all ]]; then
    Spot
  fi
done

