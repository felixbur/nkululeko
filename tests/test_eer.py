#!/usr/bin/env python3
"""
Simple test for EER implementation.
This script tests the equal_error_rate function with synthetic data.
"""

import numpy as np
import sys
from pathlib import Path

# Add nkululeko to path
nkululeko_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(nkululeko_root))

from nkululeko.reporting.reporter import equal_error_rate

def test_eer():
    """Test EER calculation with known data."""
    print("Testing Equal Error Rate implementation...")
    
    # Test case 1: Perfect classification
    print("\nTest 1: Perfect classification (EER should be ~0)")
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.15, 0.05, 0.1, 0.9, 0.95, 0.85, 0.92, 0.88])
    eer = equal_error_rate(y_true, y_score)
    print(f"  EER = {eer:.4f}")
    assert eer < 0.1, "EER should be very low for perfect classification"
    
    # Test case 2: Random classification
    print("\nTest 2: Random-like classification (EER should be ~0.5)")
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_score = np.random.rand(100)
    eer = equal_error_rate(y_true, y_score)
    print(f"  EER = {eer:.4f}")
    assert 0.3 < eer < 0.7, "EER should be around 0.5 for random classification"
    
    # Test case 3: Moderate classification
    print("\nTest 3: Moderate classification")
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 5)
    y_score = np.array([0.3, 0.4, 0.2, 0.35, 0.25, 0.7, 0.65, 0.75, 0.8, 0.68] * 5)
    eer = equal_error_rate(y_true, y_score)
    print(f"  EER = {eer:.4f}")
    assert 0 <= eer <= 1, "EER should be between 0 and 1 (inclusive)"
    
    print("\n✅ All EER tests passed!")

if __name__ == "__main__":
    try:
        test_eer()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
