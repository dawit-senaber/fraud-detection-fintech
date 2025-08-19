#!/usr/bin/env python3
"""
Script to create minimal test data for CI environment
"""
import pandas as pd
import os

def create_test_data():
    """Create minimal test datasets"""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Create minimal credit card test data
    cc_data = pd.DataFrame({
        'Time': [0, 86400, 172800],
        'Amount': [100.0, 1000.0, 500.0],
        'Class': [0, 1, 0],
        'V1': [0.1, -0.2, 0.3],
        'V2': [-0.5, 0.8, -1.2],
        'V3': [1.0, -0.5, 0.2]
    })
    cc_data.to_csv('data/creditcard_sample.csv', index=False)
    
    print("Created minimal test data for CI")

if __name__ == "__main__":
    create_test_data()