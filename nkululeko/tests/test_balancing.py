#!/usr/bin/env python3
"""
Simple and comprehensive test suite for all balancing methods in DataBalancer.

Tests all 11 balancing methods from balance.py:

Oversampling (5): ros, smote, adasyn, borderlinesmote, svmsmote
Undersampling (4): clustercentroids, randomundersampler, editednearestneighbours, tomeklinks  
Combination (2): smoteenn, smotetomek

Run with: pytest nkululeko/tests/test_balancing.py -v
"""

import numpy as np
import pandas as pd
import pytest
from nkululeko.balance import DataBalancer
import nkululeko.glob_conf as glob_conf


@pytest.fixture
def sample_data():
    """Create sample imbalanced data that works with all methods"""
    np.random.seed(42)
    
    # Majority class: 100 samples, Minority class: 25 samples
    # Well-separated for better algorithm performance
    majority_features = np.random.randn(100, 10) 
    minority_features = np.random.randn(25, 10) + 3  # Good separation
    
    features = np.vstack([majority_features, minority_features])
    labels = np.array([0] * 100 + [1] * 25)
    
    df_train = pd.DataFrame({'target': labels})
    feats_train = features
    
    return df_train, feats_train


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    original_config = getattr(glob_conf, 'config', None)
    
    glob_conf.config = {
        'FEATS': {'balancing': 'smote'},
        'DATA': {'target': 'target'},
        'MODEL': {'type': 'mlp'}
    }
    
    yield glob_conf.config
    
    if original_config is not None:
        glob_conf.config = original_config


class TestDataBalancer:
    """Simple test suite for DataBalancer - tests all 11 methods"""
    
    def test_initialization(self):
        """Test 1: DataBalancer can be initialized"""
        balancer = DataBalancer(random_state=42)
        assert balancer is not None
        assert balancer.random_state == 42
    
    def test_get_all_supported_methods(self):
        """Test 2: All 11 methods are reported as supported"""
        balancer = DataBalancer()
        methods = balancer.get_supported_methods()
        
        # Check we have all 3 categories
        assert 'oversampling' in methods
        assert 'undersampling' in methods  
        assert 'combination' in methods
        
        # Check exact counts
        assert len(methods['oversampling']) == 5
        assert len(methods['undersampling']) == 4
        assert len(methods['combination']) == 2
        
        # Total should be 11
        total = (len(methods['oversampling']) + 
                len(methods['undersampling']) + 
                len(methods['combination']))
        assert total == 11
    
    def test_method_validation(self):
        """Test 3: Method validation works correctly"""
        balancer = DataBalancer()
        
        # Valid methods
        assert balancer.is_valid_method('ros') == True
        assert balancer.is_valid_method('smote') == True
        assert balancer.is_valid_method('clustercentroids') == True
        assert balancer.is_valid_method('smoteenn') == True
        
        # Invalid methods
        assert balancer.is_valid_method('invalid') == False
        assert balancer.is_valid_method('') == False
    
    def test_all_oversampling_methods(self, sample_data, mock_config):
        """Test 4: All 5 oversampling methods work"""
        df_train, feats_train = sample_data
        balancer = DataBalancer(random_state=42)
        
        oversampling_methods = ['ros', 'smote', 'adasyn', 'borderlinesmote', 'svmsmote']
        
        for method in oversampling_methods:
            print(f"Testing oversampling: {method}")
            
            balanced_df, balanced_features = balancer.balance_features(
                df_train=df_train,
                feats_train=feats_train,
                target_column='target',
                method=method
            )
            
            # Basic checks
            assert len(balanced_df) >= len(df_train), f"{method} should increase/maintain size"
            assert len(balanced_df) == len(balanced_features), f"{method} length mismatch"
            assert balanced_features.shape[1] == feats_train.shape[1], f"{method} feature dim changed"
            
            print(f"✓ {method} passed")
    
    def test_all_undersampling_methods(self, sample_data, mock_config):
        """Test 5: All 4 undersampling methods work"""
        df_train, feats_train = sample_data
        balancer = DataBalancer(random_state=42)
        
        undersampling_methods = ['clustercentroids', 'randomundersampler', 
                               'editednearestneighbours', 'tomeklinks']
        
        for method in undersampling_methods:
            print(f"Testing undersampling: {method}")
            
            balanced_df, balanced_features = balancer.balance_features(
                df_train=df_train,
                feats_train=feats_train,
                target_column='target',
                method=method
            )
            
            # Basic checks
            assert len(balanced_df) <= len(df_train), f"{method} should decrease/maintain size"
            assert len(balanced_df) == len(balanced_features), f"{method} length mismatch"
            assert balanced_features.shape[1] == feats_train.shape[1], f"{method} feature dim changed"
            
            print(f"✓ {method} passed")
    
    def test_all_combination_methods(self, sample_data, mock_config):
        """Test 6: All 2 combination methods work"""
        df_train, feats_train = sample_data
        balancer = DataBalancer(random_state=42)
        
        combination_methods = ['smoteenn', 'smotetomek']
        
        for method in combination_methods:
            print(f"Testing combination: {method}")
            
            balanced_df, balanced_features = balancer.balance_features(
                df_train=df_train,
                feats_train=feats_train,
                target_column='target',
                method=method
            )
            
            # Basic checks
            assert len(balanced_df) == len(balanced_features), f"{method} length mismatch"
            assert balanced_features.shape[1] == feats_train.shape[1], f"{method} feature dim changed"
            assert len(balanced_df) > 0, f"{method} resulted in empty dataset"
            
            print(f"✓ {method} passed")
    
    def test_all_11_methods_comprehensive(self, sample_data, mock_config):
        """Test 7: All 11 methods work in one comprehensive test"""
        df_train, feats_train = sample_data
        balancer = DataBalancer(random_state=42)
        
        # Get all methods from the balancer itself
        all_methods = balancer.get_supported_methods()
        
        successful_methods = []
        failed_methods = []
        
        print("Testing all 11 balancing methods...")
        
        for category, methods in all_methods.items():
            for method in methods:
                try:
                    balanced_df, balanced_features = balancer.balance_features(
                        df_train=df_train,
                        feats_train=feats_train,
                        target_column='target',
                        method=method
                    )
                    
                    # Verify results
                    assert len(balanced_df) == len(balanced_features)
                    assert balanced_features.shape[1] == feats_train.shape[1]
                    assert len(balanced_df) > 0
                    
                    successful_methods.append(method)
                    print(f"✓ {method} succeeded")
                    
                except Exception as e:
                    failed_methods.append((method, str(e)))
                    print(f"✗ {method} failed: {str(e)}")
        
        print(f"\nResults: {len(successful_methods)}/11 methods successful")
        print(f"Successful: {successful_methods}")
        if failed_methods:
            print(f"Failed: {[m[0] for m in failed_methods]}")
        
        # All 11 methods should work
        assert len(successful_methods) == 11, f"Expected 11 successful methods, got {len(successful_methods)}"
        assert len(failed_methods) == 0, f"Some methods failed: {failed_methods}"
    
    def test_invalid_method_handling(self, sample_data, mock_config):
        """Test 8: Invalid methods are handled correctly"""
        df_train, feats_train = sample_data
        balancer = DataBalancer(random_state=42)
        
        # Test that invalid methods are detected by validation
        assert balancer.is_valid_method('invalid_method') == False
        assert balancer.is_valid_method('nonexistent') == False
        assert balancer.is_valid_method('') == False
        
        # Note: The actual balance_features() with invalid method calls sys.exit()
        # This is expected behavior in the current implementation
        print("✓ Invalid method validation works correctly")


def test_simple_integration():
    """Test 9: Simple integration test without fixtures"""
    print("Simple integration test...")
    
    # Create simple data
    np.random.seed(42)
    features = np.random.randn(60, 5)
    labels = np.array([0] * 40 + [1] * 20)  # 40 vs 20 imbalance
    
    df_train = pd.DataFrame({'target': labels})
    
    # Test a few key methods
    balancer = DataBalancer(random_state=42)
    key_methods = ['ros', 'smote', 'clustercentroids', 'randomundersampler']
    
    for method in key_methods:
        balanced_df, balanced_features = balancer.balance_features(
            df_train=df_train,
            feats_train=features,
            target_column='target',
            method=method
        )
        
        assert len(balanced_df) == len(balanced_features)
        print(f"✓ {method} integration test passed")
    
    print("✓ Integration test completed")


if __name__ == "__main__":
    print("Running simple balancing tests...")
    print("=" * 50)
    
    # Run integration test
    test_simple_integration()
    
    print("=" * 50)
    print("Direct test completed! Run 'pytest test_balancing.py -v' for full tests")
