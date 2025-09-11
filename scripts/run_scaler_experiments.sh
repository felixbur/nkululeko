#!/bin/bash

# Script to run all scaling method experiments
# Created for testing different scaling techniques in nkululeko
# Auto-detects if run from scripts/ or nkululeko root directory

echo "Starting scaling experiments..."
echo "==============================="

# Auto-detect working directory and set paths accordingly
CURRENT_DIR="$(pwd)"
if [[ "$CURRENT_DIR" == *"/scripts" ]]; then
    # Running from scripts/ directory
    EXAMPLES_PATH="../examples"
    RESULTS_PATH="../examples/results"
    NKULULEKO_ROOT=".."
else
    # Running from nkululeko root or elsewhere
    EXAMPLES_PATH="./examples"
    RESULTS_PATH="./examples/results"
    NKULULEKO_ROOT="."
fi

echo "Current directory: $CURRENT_DIR"
echo "Examples path: $EXAMPLES_PATH"
echo "Results path: $RESULTS_PATH"
echo ""

# List of scaling methods to test (from scaler.py)
scaling_methods=(
    "standard"
    "robust"
    "minmax"
    "maxabs"
    "normalizer"
    "powertransformer"
    "quantiletransformer"
    "bins"
    "speaker"
)

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_PATH"

# Create temporary config files directory
temp_config_dir="$RESULTS_PATH/temp_scaling_configs"
mkdir -p "$temp_config_dir"

# Initialize summary file
summary_file="$RESULTS_PATH/scaling_experiments_summary.txt"
echo "Scaling Experiments Summary" > $summary_file
echo "===========================" >> $summary_file
echo "Started: $(date)" >> $summary_file
echo "Dataset: Polish emotion data" >> $summary_file
echo "Feature type: OpenSMILE (os)" >> $summary_file
echo "Model: XGBoost" >> $summary_file
echo "" >> $summary_file

success_count=0
total_count=${#scaling_methods[@]}

# Function to create config file for a scaling method
create_scaling_config() {
    local method=$1
    local config_file="$temp_config_dir/exp_scaling_${method}.ini"
    
    cat > "$config_file" << EOF
[EXP]
root = $RESULTS_PATH/
name = exp_polish_scaling_${method}
runs = 1
epochs = 5
save = True

[DATA]
databases = ['train', 'dev', 'test']
train = ./data/polish/polish_train.csv
train.type = csv
train.absolute_path = False
train.split_strategy = train
dev = ./data/polish/polish_dev.csv
dev.type = csv
dev.absolute_path = False
dev.split_strategy = train
test = ./data/polish/polish_test.csv
test.type = csv
test.absolute_path = False
test.split_strategy = test
target = emotion

[FEATS]
type = ['os']
scale = ${method}

[MODEL]
type = xgb
save = True

[PLOT]
name = ${method}_scaling_results
EOF

    echo "$config_file"
}

# Function to check if Polish data exists
check_polish_data() {
    local train_file="$NKULULEKO_ROOT/data/polish/polish_train.csv"
    local dev_file="$NKULULEKO_ROOT/data/polish/polish_dev.csv"
    local test_file="$NKULULEKO_ROOT/data/polish/polish_test.csv"
    
    if [[ ! -f "$train_file" || ! -f "$dev_file" || ! -f "$test_file" ]]; then
        echo "WARNING: Polish dataset not found at expected location:"
        echo "  Expected: $NKULULEKO_ROOT/data/polish/"
        echo "  Files needed: polish_train.csv, polish_dev.csv, polish_test.csv"
        echo ""
        echo "Using test dataset instead..."
        return 1
    fi
    return 0
}

# Function to create fallback config with test dataset
create_fallback_config() {
    local method=$1
    local config_file="$temp_config_dir/exp_scaling_${method}.ini"
    
    cat > "$config_file" << EOF
[EXP]
root = $RESULTS_PATH/
name = exp_test_scaling_${method}
runs = 1
epochs = 3
save = True

[DATA]
databases = ['testdb']
testdb = ./data/test/samples.csv
testdb.type = csv
testdb.absolute_path = False
testdb.split_strategy = train_test
target = emotion

[FEATS]
type = ['os']
scale = ${method}

[MODEL]
type = xgb
save = True

[PLOT]
name = ${method}_scaling_results
EOF

    echo "$config_file"
}

# Check data availability and inform user
echo "Checking data availability..."
if check_polish_data; then
    echo "✓ Polish dataset found - using full dataset"
    data_type="polish"
else
    echo "ℹ Using test dataset as fallback"
    data_type="test"
fi
echo ""

# Run each experiment
for method in "${scaling_methods[@]}"; do
    echo ""
    echo "Running experiment with scaling method: $method"
    echo "================================================="
    
    # Create appropriate config file
    if [[ "$data_type" == "polish" ]]; then
        config_file=$(create_scaling_config "$method")
    else
        config_file=$(create_fallback_config "$method")
    fi
    
    log_file="$RESULTS_PATH/exp_scaling_${method}.log"
    
    echo "Config file created: $config_file"
    echo "Starting experiment..."
    
    # Run the experiment and capture output
    if cd "$NKULULEKO_ROOT" && python3 -m nkululeko.nkululeko --config "$config_file" > "$log_file" 2>&1; then
        echo "✓ SUCCESS: $method scaling completed"
        echo "✓ SUCCESS: $method" >> $summary_file
        ((success_count++))
        
        # Extract result from log if available
        if grep -q "best result" "$log_file"; then
            result=$(grep "best result" "$log_file" | tail -1)
            echo "  Result: $result"
            echo "  Result: $result" >> $summary_file
        fi
    else
        echo "✗ FAILED: $method scaling failed"
        echo "✗ FAILED: $method (see log: $log_file)" >> $summary_file
        echo "Error details:"
        tail -20 "$log_file"
    fi
    
    echo "Log saved to: $log_file"
done

# Final summary
echo ""
echo "========================================"
echo "All scaling experiments completed!"
echo "Success: $success_count/$total_count"
echo "========================================"

echo "" >> $summary_file
echo "Completed: $(date)" >> $summary_file
echo "Success rate: $success_count/$total_count" >> $summary_file

# Add scaling method descriptions to summary
echo "" >> $summary_file
echo "Scaling Methods Tested:" >> $summary_file
echo "======================" >> $summary_file
echo "standard:            Z-score normalization (mean=0, std=1)" >> $summary_file
echo "robust:              Median and IQR based scaling (outlier resistant)" >> $summary_file
echo "minmax:              Scale to range [0,1]" >> $summary_file
echo "maxabs:              Scale by maximum absolute value" >> $summary_file
echo "normalizer:          L2 normalization (unit norm per sample)" >> $summary_file
echo "powertransformer:    Gaussian-like transformation" >> $summary_file
echo "quantiletransformer: Map to uniform/Gaussian distribution" >> $summary_file
echo "bins:                Convert to 3 categorical bins" >> $summary_file
echo "speaker:             Per-speaker standard scaling" >> $summary_file

echo "Summary saved to: $summary_file"

# Cleanup temporary config files
echo ""
echo "Cleaning up temporary config files..."
rm -rf "$temp_config_dir"
echo "Temporary files removed."

# Provide next steps
echo ""
echo "Next Steps:"
echo "==========="
echo "1. Check the summary file: $summary_file"
echo "2. Review individual log files for detailed results"
echo "3. Compare results to choose the best scaling method for your data"
echo "4. Use the best performing scaling method in your final experiments"

# Show quick comparison if we can extract results
echo ""
echo "Quick Results Comparison:"
echo "========================"
for method in "${scaling_methods[@]}"; do
    log_file="$RESULTS_PATH/exp_scaling_${method}.log"
    if [[ -f "$log_file" ]] && grep -q "UAR:" "$log_file"; then
        result=$(grep "UAR:" "$log_file" | tail -1 | awk '{print $NF}')
        printf "%-20s: %s\n" "$method" "$result"
    elif [[ -f "$log_file" ]] && grep -q "ERROR" "$log_file"; then
        printf "%-20s: FAILED\n" "$method"
    else
        printf "%-20s: No result found\n" "$method"
    fi
done
