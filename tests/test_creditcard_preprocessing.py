import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import using proper Python module syntax
from src.preprocessing.creditcard_preprocessing import create_preprocessing_pipeline, validate_data

def test_data_validation():
    """Test valid data structure"""
    data = pd.DataFrame({
        'Time': [1, 2, 3],
        'Amount': [100, 200, 300],
        'Class': [0, 1, 0],
        'V1': [0.1, -0.2, 0.3]
    })
    assert validate_data(data) is True

def test_validate_data_missing_column():
    """Test missing required column"""
    data = pd.DataFrame({'Time': [1, 2], 'Amount': [100, 200]})
    with pytest.raises(ValueError):
        validate_data(data)

def test_preprocessing_pipeline_structure():
    """Test pipeline creation"""
    pipeline = create_preprocessing_pipeline()
    assert hasattr(pipeline, 'fit_transform')
    assert hasattr(pipeline, 'transform')

def test_preprocessing_pipeline_transform():
    """Test pipeline transformation"""
    pipeline = create_preprocessing_pipeline()
    data = pd.DataFrame({
        'Time': [0, 86400], 
        'Amount': [10, 1000],
        'Class': [0, 1],
        'V1': [0.1, -0.2],
        'V2': [-0.5, 0.8]
    })
    transformed = pipeline.fit_transform(data)
    assert transformed.shape[0] == 2  # Same number of rows
    assert not np.isnan(transformed).any()  # No NaN values