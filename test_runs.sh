# test explore module
python -m nkululeko.explore --config tests/exp_emodb_explore_data.ini
python -m nkululeko.explore --config tests/exp_emodb_explore_featimportance.ini
python -m nkululeko.explore --config tests/exp_emodb_explore_scatter.ini
python -m nkululeko.explore --config tests/exp_emodb_explore_features.ini
python -m nkululeko.explore --config tests/exp_androids_explore.ini
python -m nkululeko.explore --config tests/exp_agedb_explore_data.ini
# test basic nkululeko
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
# test augmentation
python -m nkululeko.augment --config tests/exp_emodb_augment_os_xgb.ini
python -m nkululeko.nkululeko --config tests/exp_emodb-aug_os_xgb.ini
python -m nkululeko.augment --config tests/exp_emodb_random_splice_os_xgb.ini
python -m nkululeko.nkululeko --config tests/exp_emodb_rs_os_xgb.ini
python -m nkululeko.aug_train --config tests/emodb_aug_train.ini
python -m nkululeko.nkululeko --config tests/exp_androids_os_svm.ini
# test prediction
python -m nkululeko.predict --config tests/exp_emodb_predict.ini
python -m nkululeko.nkululeko --config tests/emodb_demo.ini
# test demo
python -m nkululeko.demo --config tests/emodb_demo.ini --list data/test/samples.csv
# test test module
python -m nkululeko.nkululeko --config exp_emodb_os_xgb_test.ini
python -m nkululeko.test --config exp_emodb_os_xgb_test.ini
# test multidb
python -m nkululeko.multidb --config tests/exp_multidb.ini
# test spotlight
python -m nkululeko.explore --config tests/exp_explore.ini
