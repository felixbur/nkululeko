#!/bin/bash

# script to run all example configurations in the nkululeko package
example_dir="examples"

# save output to log.out
exec > >(tee -i log.out)

# save error to log.err
exec 2> >(tee -i log.err >&2)


# Display help message
function Help {
    echo "Usage: test_runs.sh [options]"
    echo "Example: ./run_examples.sh nkululeko"
    echo "Options:"
    echo "  nkululeko: test basic nkululeko"
    echo "  augment: test augmentation"
    echo "  predict: test the unified predict module with autopredict targets / feature extractors"
    echo "  demo: test predict --type model on classification configs (replaces the old nkululeko.demo)"
    echo "  testing: test predict --type model on saved-test-set configs (replaces the old nkululeko.testing)"
    echo "  multidb: test multidb"
    echo "  explore: test explore module (must be run last)"
    echo "  all: test all modules"
    echo "  -spotlight: test all modules except spotlight (useful in SSH)"
    echo "  -overwrite: remove (old) results directory and create if not exist"
    echo "  --help: display this help message"
}

# rm results dir if argument is "nkululeko" or "all"
# TODO: move root to /tmp so no need to do this
if [ "$1" == "nkululeko" ] || [ "$1" == "all" ];  [ "$1" == "-spotlight" ]; then 
    # add overwrite argument
    if [ "$2" == "-overwrite" ]; then
        echo "Removing (old) results directory and create if not exist"
        rm -rf $example_dir/results/*
        mkdir -p $example_dir/results
    fi
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

# resample before performing other tests
resample_ini_files=(
    exp_polish_gmm.ini
)
# test basic nkululeko
nkululeko_ini_files=(
    exp_emodb_os_praat_xgb.ini
    exp_emodb_limit_size.ini
    exp_emodb_featimport_xgb.ini
    exp_emodb_mapping.ini
    exp_emodb_cnn.ini
    exp_emodb_balancing.ini
    exp_emodb_split.ini
    exp_emodb_traindevtest.ini
    exp_emodb_traindevtest_split.ini
    exp_emodb_filter.ini
    exp_emodb_stress.ini
    exp_emodb_binscaled.ini
    exp_ravdess_os_xgb.ini
    exp_agedb_class_os_xgb.ini
    exp_emodb_hubert_xgb.ini
    exp_emodb_wavlm_xgb.ini
    exp_emodb_whisper_xgb.ini
    emodb_demo.ini
    exp_emodb_stratify.ini
    exp_emodb_os_xgb_test.ini
    exp_emodb_wav2vec2_test.ini
    exp_emodb_audmodel_mlp.ini
    exp_emodb_os_xgb.ini
    exp_emodb_nospeaker.ini
    exp_emodb_os_svm.ini
    exp_emodb_sptk_svm.ini
    exp_emodb_os_knn.ini
    exp_emodb_os_mlp.ini
    exp_agedb_os_xgr.ini
    exp_agedb_os_mlp.ini
    exp_polish_gmm.ini
    exp_emodb-aug_os_xgb.ini
    exp_multi_testsets_neutral_angry.ini
)

# test augmentation
augment_ini_files=(
    augment_auglib.ini
    exp_emodb_augment_os_xgb.ini
    exp_emodb_random_splice_os_xgb.ini
    exp_emodb_rs_os_xgb.ini
    emodb_aug_train.ini
)

# test predict: exercise the new unified predict module on data/test/samples.csv.
# Each entry is passed as --model <NAME> to nkululeko.predict.
# Names matching AUTOPREDICT_TARGETS (e.g. snr) hit the autopredict path;
# others (e.g. praat, opensmile) are treated as feature-extractor names.
# Kept short to keep the smoke fast and avoid heavyweight model downloads.
predict_ini_files=(
    snr
    praat
    opensmile
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
    exp_polish_gmm.ini  # shap
)

ensemble_ini_files=(
    exp_emodb_os_knn.ini
    exp_emodb_os_svm.ini
)

if [ $# -eq 0 ] || [ "$1" == "--help" ]; then
    Help
fi

start_time=$(date +%s)

# Loop over the module or all modules if -all arg is given
if [ "$1" == "all" ]; then
    modules=(nkululeko augment predict demo testing multidb explore)
elif [ "$1" == "-spotlight" ]; then
    modules=(resample nkululeko augment predict demo testing multidb explore)
    # unset last two ini files to exclude spotlight and shap
    unset explore_ini_files[-1]  # Exclude INI file for spotlight 
    unset explore_ini_files[-1]  # and shap
else
    modules=("$@")
fi

success_count=0
failed_count=0
for module in "${modules[@]}"
do
    # Resolve which list of entries to iterate over for this module.
    # `testing` reuses the test_ini_files array; everything else uses
    # `<module>_ini_files`.
    if [ "$module" == "testing" ]; then
        ini_files="test_ini_files[@]"
    else
        ini_files="${module}_ini_files[@]"
    fi

    for ini_file in "${!ini_files}"
    do
        # demo and testing both run the new predict module in model mode
        # against a small list of audio samples.
        if [ "$module" == "demo" ] || [ "$module" == "testing" ]; then
            outfile="/tmp/${module}_${ini_file%.ini}_pred.csv"
            RunTest python3 -m nkululeko.predict \
                --type model \
                --config "$example_dir/$ini_file" \
                --list "data/test/samples.csv" \
                --outfile "$outfile"

        # The new predict module is driven directly via CLI args, not by a
        # [PREDICT] section. Each entry in predict_ini_files is a model
        # name (autopredict target or feature-extractor name).
        elif [ "$module" == "predict" ]; then
            outfile="/tmp/predict_${ini_file}.csv"
            RunTest python3 -m nkululeko.predict \
                --list "data/test/samples.csv" \
                --model "$ini_file" \
                --outfile "$outfile"

        # for ensemble module
        elif [ "$module" == "ensemble" ]; then
            # combine all ini files
            inis = ""
            for ensemble_ini_file in "${ensemble_ini_files[@]}"
            do
                inis += "$example_dir/$ensemble_ini_file "
            done
            RunTest python3 -m "nkululeko.$module" $inis --method mean

        else # for other modules
            RunTest python3 -m "nkululeko.$module" --config "$example_dir/$ini_file"
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
