import pytest
import pandas as pd
import numpy as np
from src/preprocessing/creditcard_preprocessing import create_preprocessing_pipeline, validate_data
from sklearn.pipeline import Pipeline

def test_validate_data_valid():
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
    assert isinstance(pipeline, Pipeline)
    assert 'time_amt' in pipeline.named_transformers_
    assert 'log_amt' in pipeline.named_transformers_

def test_preprocessing_pipeline_transform():
    """Test pipeline transformation"""
    pipeline = create_preprocessing_pipeline()
    data = pd.DataFrame({'Time': [0, 86400], 'Amount': [10, 1000]})
    transformed = pipeline.fit_transform(data)
    assert transformed.shape == (2, 3)
    assert not np.isnan(transformed).any()