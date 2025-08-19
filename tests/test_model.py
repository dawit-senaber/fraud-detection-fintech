# tests/test_model.py
import pytest
import joblib
from src.model_training import train_and_evaluate
import pandas as pd
import numpy as np
import os

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

    # Create minimal config for testing
    config = {
        'financial_impact': {
            'false_positive_cost': 10,
            'false_negative_cost': 100
        },
        'results': {
            'cost_dir': str(tmp_path / 'cost_curves')
        },
        'models_dir': str(tmp_path / 'models')  # Added models_dir
    }

    # Test training - changed dataset_name to 'creditcard'
    result = train_and_evaluate(
        dataset_name='creditcard',
        model_type='logistic',
        train_path=str(train_path),
        test_path=str(test_path),
        target_col='Class',
        config=config
    )

    # Verify outputs
    assert 'metrics' in result
    assert os.path.exists(result['model_path'])
    # For minimal test data, we can't expect high performance
    assert 'auc_roc' in result['metrics']