import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_import_preprocessing():
    """Test that we can import preprocessing modules"""
    try:
        from src.preprocessing import creditcard_preprocessing
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import preprocessing module: {e}")

def test_dummy():
    """Simple test to ensure framework works"""
    assert 1 + 1 == 2