"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
from typing import Generator, Dict, Any

import pandas as pd
import numpy as np
from src.config import AppConfig, get_config
from src.persistence import ModelStorage


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def test_config(temp_dir: Path) -> AppConfig:
    """Create a test configuration."""
    config = AppConfig()
    config.storage.model_storage_path = str(temp_dir / "test_models")
    config.storage.data_storage_path = str(temp_dir / "test_data")
    return config


@pytest.fixture
def sample_data() -> Dict[str, Any]:
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Create sample features
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randint(0, 5, n_samples),
        'feature4': np.random.uniform(0, 10, n_samples)
    })
    
    # Create sample target (binary classification)
    y = pd.Series(np.random.randint(0, 2, n_samples), name='target')
    
    return {
        'X': X,
        'y': y,
        'feature_names': list(X.columns),
        'target_name': 'target'
    }


@pytest.fixture
def sample_regression_data() -> Dict[str, Any]:
    """Create sample regression data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Create sample features
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.uniform(0, 10, n_samples)
    })
    
    # Create sample target (regression)
    y = pd.Series(
        2 * X['feature1'] + 3 * X['feature2'] + np.random.randn(n_samples) * 0.1,
        name='target'
    )
    
    return {
        'X': X,
        'y': y,
        'feature_names': list(X.columns),
        'target_name': 'target'
    }


@pytest.fixture
def model_storage(test_config: AppConfig) -> ModelStorage:
    """Create a ModelStorage instance for testing."""
    return ModelStorage(test_config)


@pytest.fixture
def mock_model_data() -> Dict[str, Any]:
    """Create mock model data for persistence tests."""
    return {
        'model_name': 'test_model',
        'model_version': '1.0.0',
        'algorithm': 'xgboost',
        'parameters': {
            'max_depth': 6,
            'learning_rate': 0.3,
            'n_estimators': 100
        },
        'feature_names': ['feature1', 'feature2', 'feature3'],
        'target_name': 'target',
        'model_type': 'classification'
    } 