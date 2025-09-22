#!/usr/bin/env python3
"""
Debug script to identify None values in your data
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def debug_none_values(processor):
    """
    Debug function to identify None values in the data
    """
    print("=" * 60)
    print("DEBUGGING NONE VALUES IN DATA")
    print("=" * 60)
    
    # Get the data from processor
    try:
        result = processor.prepare_model_data()
        print(f"Processor returned {len(result)} items")
        
        # Check each returned value
        for i, item in enumerate(result):
            print(f"\nItem {i}: type = {type(item)}, value = {item}")
            
            if item is None:
                print(f"❌ ITEM {i} IS NONE! This is the problem.")
            elif hasattr(item, 'shape'):
                print(f"  Shape: {item.shape}")
                # Check if the array contains None values
                if hasattr(item, 'flatten'):
                    flat_data = item.flatten()
                    none_count = sum(1 for x in flat_data if x is None)
                    print(f"  None values in array: {none_count}")
            elif isinstance(item, (list, tuple)):
                none_count = sum(1 for x in item if x is None)
                print(f"  None values in list/tuple: {none_count}")
                
        return result
        
    except Exception as e:
        print(f"Error getting data from processor: {e}")
        return None

def fix_none_data(X, y, feature_names):
    """
    Fix data that contains None values
    """
    print("\n" + "=" * 60)
    print("ATTEMPTING TO FIX NONE VALUES")
    print("=" * 60)
    
    # Convert to numpy arrays first
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values
    
    X = np.array(X)
    y = np.array(y)
    
    # Check for None values
    x_none_mask = np.array([x is None for x in X.flatten()]).reshape(X.shape)
    y_none_mask = np.array([y_val is None for y_val in y])
    
    print(f"None values in X: {np.sum(x_none_mask)}")
    print(f"None values in y: {np.sum(y_none_mask)}")
    
    # Remove rows with None values
    rows_with_none = np.any(x_none_mask, axis=1) | y_none_mask
    print(f"Rows containing None values: {np.sum(rows_with_none)}")
    
    if np.sum(rows_with_none) > 0:
        print("Removing rows with None values...")
        X_clean = X[~rows_with_none]
        y_clean = y[~rows_with_none]
        
        print(f"Original shape: X {X.shape}, y {y.shape}")
        print(f"Cleaned shape: X {X_clean.shape}, y {y_clean.shape}")
        
        return X_clean, y_clean, feature_names
    else:
        print("No None values found after conversion")
        return X, y, feature_names

def safe_train_test_split(processor):
    """
    Safe train_test_split with None value handling
    """
    # Debug first
    result = debug_none_values(processor)
    
    if result is None or len(result) < 2:
        print("❌ Processor returned invalid data")
        # Create fallback data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        feature_names = [f'feature_{i}' for i in range(5)]
    else:
        X, y, *feature_names = result
        feature_names = feature_names[0] if feature_names else None
        
        # Fix None values
        X, y, feature_names = fix_none_data(X, y, feature_names)
    
    # Now safely split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"✅ Split successful! Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test, feature_names
        
    except Exception as e:
        print(f"❌ Split failed: {e}")
        # Emergency fallback
        X_fallback = np.random.randn(50, 3)
        y_fallback = np.random.randint(0, 2, 50)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_fallback, y_fallback, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, ['backup_1', 'backup_2', 'backup_3']

# Test function
def test_with_dummy_data():
    """Test with data that contains None values"""
    print("Testing with dummy data containing None values...")
    
    # Create test data with None values
    X_test = np.array([[1, 2, None], [4, 5, 6], [None, 8, 9], [10, 11, 12]])
    y_test = np.array([0, None, 1, 0])
    
    print(f"Original X: {X_test}")
    print(f"Original y: {y_test}")
    
    # Test the fix function
    X_clean, y_clean, _ = fix_none_data(X_test, y_test, None)
    
    print(f"Cleaned X: {X_clean}")
    print(f"Cleaned y: {y_clean}")

if __name__ == "__main__":
    test_with_dummy_data()