import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test that basic imports work"""
    try:
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import required packages: {e}")


def test_dummy():
    """Simple test to ensure framework works"""
    assert 1 + 1 == 2
