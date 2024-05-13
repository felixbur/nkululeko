#!/bin/bash

# directory for test files
# test_dir="tests"

# Display help message
function Help {
    echo "Usage: test_runs.sh [options]"
    echo "Example: ./test_runs.sh nkululeko"
    echo "Options:"
    echo "  nkululeko: test basic nkululeko"
    echo "  augment: test augmentation"
    echo "  predict: test prediction"
    echo "  demo: test demo"
    echo "  test: test test module"
    echo "  multidb: test multidb"
    echo "  explore: test explore module (must be run last)"
    echo "  all: test all modules"
    echo "  -spotlight: test all modules except spotlight (useful in SSH)"
    echo "  --help: display this help message"
}

# rm results dir if argument is "nkululeko" or "all"
# TODO: move root to /tmp so no need to do this
if [ "$1" == "nkululeko" ] || [ "$1" == "all" ];  [ "$1" == "-spotlight" ]; then
    echo "Removing (old) results directory and create if not exist"
    rm -rf tests/results/*
    mkdir -p tests/results
fi

# Run a test and check for errors
function RunTest {
    "$@"
    if [ $? -ne 0 ]; then
        echo "Error: Test failed - $@"
        return 1   # exit after error message
    # else
    #     return 0   # continue after error message
    fi
}

# test basic nkululeko
nkululeko_ini_files=(
    exp_emodb_os_praat_xgb.ini
    exp_emodb_featimport_xgb.ini
    exp_emodb_cnn.ini
    exp_emodb_balancing.ini
    exp_emodb_split.ini
    exp_ravdess_os_xgb.ini
    exp_agedb_class_os_xgb.ini
    exp_emodb_hubert_xgb.ini
    exp_emodb_wavlm_xgb.ini
    exp_emodb_whisper_xgb.ini
    emodb_demo.ini
    exp_emodb_os_xgb_test.ini
    exp_emodb_trill_test.ini
    exp_emodb_wav2vec2_test.ini
    exp_emodb_os_xgb.ini
    exp_emodb_os_svm.ini
    exp_emodb_os_knn.ini
    exp_emodb_os_mlp.ini
    exp_agedb_os_xgr.ini
    exp_agedb_os_mlp.ini
)

# test augmentation
augment_ini_files=(
    exp_emodb_augment_os_xgb.ini
    exp_emodb-aug_os_xgb.ini
    exp_emodb_random_splice_os_xgb.ini
    exp_emodb_rs_os_xgb.ini
    emodb_aug_train.ini
)

# test prediction
predict_ini_files=(
    exp_emodb_predict.ini
) 
# test demo
demo_ini_files=(
    exp_emodb_os_xgb.ini
    exp_emodb_os_svm.ini
    exp_emodb_os_knn.ini
    exp_emodb_os_mlp.ini
    exp_agedb_os_xgr.ini
    exp_agedb_os_mlp.ini
)

# test test module
test_ini_files=(
    exp_emodb_os_xgb_test.ini
    exp_emodb_trill_test.ini
    exp_emodb_wav2vec2_test.ini
)

# test multidb
multidb_ini_files=(
    exp_multidb.ini
)

# test explore module
explore_ini_files=(
    exp_emodb_explore_data.ini
    exp_emodb_explore_featimportance.ini 
    exp_emodb_explore_scatter.ini
    exp_emodb_explore_features.ini
    exp_agedb_explore_data.ini
    exp_explore.ini # test splotlight
)


if [ $# -eq 0 ] || [ "$1" == "--help" ]; then
    Help
fi

start_time=$(date +%s)

# Loop over the module or all modules if -all arg is given
if [ "$1" == "all" ]; then
    modules=(nkululeko augment predict demo test multidb explore)
elif [ "$1" == "-spotlight" ]; then
    modules=(nkululeko augment predict demo test multidb explore)
    unset explore_ini_files[-1]  # Exclude INI file for spotlight
else
    modules=("$@")
fi

success_count=0
failed_count=0
for module in "${modules[@]}"
do
    # Run the test over the selected modules
    ini_files="${module}_ini_files[@]"
    for ini_file in "${!ini_files}"
    do
        # if module is "demo" add "--list" argument
        if [ "$module" == "demo" ]; then
            RunTest python3 -m "nkululeko.$module" --config "tests/$ini_file" --list "data/test/samples.csv"
        else
            RunTest python3 -m "nkululeko.$module" --config "tests/$ini_file"
        fi

        if [ $? -eq 0 ]; then
            ((success_count++))
        else
            ((failed_count++))
            failed_modules+=("$module with $ini_file")
        fi
    done
done

echo "Total tests passed: $success_count"
echo "Total tests failed: $failed_count"

if [ ${#failed_modules[@]} -gt 0 ]; then
    echo "Failed modules and INI files:"
    for failed_module in "${failed_modules[@]}"; do
        echo "$failed_module"
    done
fi

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo "Total time taken: $total_time seconds"
