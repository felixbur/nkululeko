#!/usr/bin/env python3
"""Test script to verify activation functions in MLP model."""

import sys
import torch
import torch.nn as nn

# Test activation functions
activation_functions = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'elu': nn.ELU(),
    'selu': nn.SELU(),
    'gelu': nn.GELU(),
}

print("Testing activation functions available in PyTorch:")
print("=" * 50)

for name, activation in activation_functions.items():
    try:
        # Test with sample input
        x = torch.randn(10, 20)
        y = activation(x)
        print(f"✓ {name:15s} - Shape: {y.shape}, Mean: {y.mean():.4f}, Std: {y.std():.4f}")
    except Exception as e:
        print(f"✗ {name:15s} - Error: {e}")
        sys.exit(1)

print("\n" + "=" * 50)
print("All activation functions work correctly!")

# Test nkululeko imports
try:
    from nkululeko.models.model_mlp import MLPModel
    print("\n✓ Successfully imported MLPModel from nkululeko")
    
    # Check if _get_activation method exists
    import inspect
    if hasattr(MLPModel, '_get_activation'):
        print("✓ _get_activation method exists in MLPModel")
        
        # Check the method signature
        sig = inspect.signature(MLPModel._get_activation)
        print(f"  Method signature: {sig}")
    else:
        print("✗ _get_activation method not found")
        sys.exit(1)
        
except Exception as e:
    print(f"\n✗ Failed to import or test MLPModel: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTest passed! ✓")
