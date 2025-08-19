import pytest
import pandas as pd
import os
from src.preprocessing import creditcard_preprocessing as ccp


def test_data_validation():
    # Valid data
    valid_data = pd.DataFrame(
        {"Time": [0, 86400], "Amount": [10, 1000], "Class": [0, 1]}
    )
    assert ccp.validate_data(valid_data) is True

    # Missing column
    invalid_data = valid_data.drop(columns=["Amount"])
    with pytest.raises(ValueError):
        ccp.validate_data(invalid_data)

    # Invalid class values
    invalid_class = valid_data.copy()
    invalid_class["Class"] = [0, 2]
    with pytest.raises(ValueError):
        ccp.validate_data(invalid_class)


def test_preprocessing_pipeline(tmp_path):
    # Create test data
    data = pd.DataFrame(
        {
            "Time": [0, 86400, None],
            "Amount": [10, 1000, 500],
            "Class": [0, 1, 0],
            "V1": [0.1, -0.2, 0.3],
        }
    )
    input_path = tmp_path / "test.csv"
    output_path = tmp_path / "processed.csv"
    data.to_csv(input_path, index=False)

    # Run preprocessing
    processed = ccp.preprocess_creditcard(str(input_path), str(output_path), {})

    # Verify results
    assert os.path.exists(output_path)
    assert len(processed) == 2  # None value removed
    assert "Hour" in processed.columns
    assert "Transaction_Value_Ratio" in processed.columns
