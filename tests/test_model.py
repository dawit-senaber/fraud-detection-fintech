import os
import pytest
import joblib
from src.model_training import train_and_evaluate
import pandas as pd
import numpy as np

def test_model_training(tmp_path):
    # Create simple dataset
    data = pd.DataFrame({
        'Time': [0, 1, 2, 3],
        'Amount': [10, 20, 30, 40],
        'Class': [0, 1, 0, 1],
        'V1': [0.1, 0.2, 0.3, 0.4]
    })
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    data.to_csv(train_path, index=False)
    data.to_csv(test_path, index=False)
    
    # Test training
    result = train_and_evaluate(
        dataset_name='test',
        model_type='logistic',
        train_path=str(train_path),
        test_path=str(test_path),
        target_col='Class'
    )
    
    # Verify outputs
    assert 'metrics' in result
    assert os.path.exists(result['model_path'])
    assert result['metrics']['auc_roc'] > 0.5