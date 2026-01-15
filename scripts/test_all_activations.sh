#!/bin/bash
# Test different activation functions

set -e

echo "Testing different activation functions..."
echo "========================================"

for activation in relu tanh sigmoid leaky_relu; do
    echo ""
    echo "Testing activation: $activation"
    echo "-----------------------------------"
    
    # Create temporary config
    cat > /tmp/test_${activation}.ini << EOF
[EXP]
root = /tmp/test_results/
name = test_${activation}
runs = 1
epochs = 3
save = False
[DATA]
databases = ['emodb']
emodb = ./data/emodb/emodb
emodb.split_strategy = specified
emodb.test_tables = ['emotion.categories.test.gold_standard']
emodb.train_tables = ['emotion.categories.train.gold_standard']
target = emotion
labels = ['anger', 'happiness']
[FEATS]
type = ['os']
scale = standard
[MODEL]
type = mlp
layers = [32, 16]
activation = ${activation}
drop = .2
patience = 2
[PLOT]
best_model = False
EOF

    # Run the test
    if timeout 45 python -m nkululeko.nkululeko --config /tmp/test_${activation}.ini > /tmp/test_${activation}.log 2>&1; then
        echo "✓ $activation: SUCCESS"
        grep "using activation function" /tmp/test_${activation}.log
        grep "DONE" /tmp/test_${activation}.log
    else
        echo "✗ $activation: FAILED"
        tail -20 /tmp/test_${activation}.log
        exit 1
    fi
done

echo ""
echo "========================================"
echo "All activation functions tested successfully!"
