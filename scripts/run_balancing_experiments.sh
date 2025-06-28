#!/bin/bash

# Script to run all balancing method experiments
# Created for testing different balancing techniques in nkululeko
# Auto-detects if run from run_scripts/ or nkululeko root directory

echo "Starting balancing experiments..."
echo "================================"

# Auto-detect working directory and set paths accordingly
CURRENT_DIR="$(pwd)"
if [[ "$CURRENT_DIR" == *"/run_scripts" ]]; then
    # Running from run_scripts/ directory
    EXAMPLES_PATH="../examples"
    RESULTS_PATH="../examples/results"
else
    # Running from nkululeko root or elsewhere
    EXAMPLES_PATH="./examples"
    RESULTS_PATH="./examples/results"
fi

echo "Current directory: $CURRENT_DIR"
echo "Examples path: $EXAMPLES_PATH"
echo "Results path: $RESULTS_PATH"
echo ""

# List of balancing methods to test
balancing_methods=(
    "ros"
    "smote" 
    "adasyn"
    "smoteenc"
    "borderlinesmote"
    "clustercentroids"
    "randomundersampler"
    "editednearestneighbours"
    "tomeklinks"
    "smotetomek"
)

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_PATH"

# Initialize summary file
summary_file="$RESULTS_PATH/balancing_experiments_summary.txt"
echo "Balancing Experiments Summary" > $summary_file
echo "=============================" >> $summary_file
echo "Started: $(date)" >> $summary_file
echo "" >> $summary_file

success_count=0
total_count=${#balancing_methods[@]}

# Run each experiment
for method in "${balancing_methods[@]}"; do
    echo ""
    echo "Running experiment with balancing method: $method"
    echo "================================================="
    
    config_file="$EXAMPLES_PATH/exp_balancing_${method}.ini"
    log_file="$RESULTS_PATH/exp_balancing_${method}.log"
    
    if [ -f "$config_file" ]; then
        echo "Config file found: $config_file"
        echo "Starting experiment..."
        
        # Run the experiment and capture output
        if python3 -m nkululeko.nkululeko --config "$config_file" > "$log_file" 2>&1; then
            echo "✓ SUCCESS: $method balancing completed"
            echo "✓ SUCCESS: $method" >> $summary_file
            ((success_count++))
        else
            echo "✗ FAILED: $method balancing failed"
            echo "✗ FAILED: $method (see log: $log_file)" >> $summary_file
            echo "Error details:"
            tail -20 "$log_file"
        fi
    else
        echo "✗ ERROR: Config file not found: $config_file"
        echo "✗ ERROR: $method - config file missing" >> $summary_file
    fi
    
    echo "Log saved to: $log_file"
done

# Final summary
echo ""
echo "========================================"
echo "All experiments completed!"
echo "Success: $success_count/$total_count"
echo "========================================"

echo "" >> $summary_file
echo "Completed: $(date)" >> $summary_file
echo "Success rate: $success_count/$total_count" >> $summary_file

echo "Summary saved to: $summary_file"

# Show results directory
echo ""
echo "Results directory contents:"
ls -la "$RESULTS_PATH"
