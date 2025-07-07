"""Tests for src/utils.py module.

This module tests all utility functions used throughout the XGBoost service
including data validation, model information formatting, feature importance
analysis, and sample data generation.
"""

import pytest
import pandas as pd
import numpy as np
import xgboost as xgb
from unittest.mock import Mock, patch, MagicMock

from src.utils import (
    validate_data,
    format_model_info,
    format_feature_importance,
    prepare_sample_data,
)


class TestValidateData:
    """Test the validate_data function."""

    def test_validate_data_valid_dataset(self):
        """Test validation of a valid dataset."""
        # Create a valid dataset
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [100, 200, 300, 400, 500]
        })
        
        result = validate_data(df, 'target')
        assert result == "valid"

    def test_validate_data_empty_dataset(self):
        """Test validation of empty dataset."""
        df = pd.DataFrame()
        
        result = validate_data(df, 'target')
        assert result == "Dataset is empty"

    def test_validate_data_missing_target_column(self):
        """Test validation when target column is missing."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        result = validate_data(df, 'target')
        assert result == "Target column 'target' not found in data"

    def test_validate_data_target_all_null(self):
        """Test validation when target column contains only null values."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'target': [None, None, None]
        })
        
        result = validate_data(df, 'target')
        assert result == "Target column 'target' contains only null values"

    def test_validate_data_excessive_null_values(self):
        """Test validation when columns have >50% null values."""
        df = pd.DataFrame({
            'feature1': [1, 2, None, None, None],  # 60% null
            'feature2': [10, 20, 30, 40, 50],
            'target': [100, 200, 300, 400, 500]
        })
        
        result = validate_data(df, 'target')
        assert "Columns with >50% null values" in result
        assert "feature1" in result

    def test_validate_data_non_numeric_columns(self):
        """Test validation when non-numeric columns are present."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': ['a', 'b', 'c'],  # Non-numeric
            'target': [100, 200, 300]
        })
        
        result = validate_data(df, 'target')
        assert "Non-numeric columns detected" in result
        assert "feature2" in result

    def test_validate_data_non_numeric_target_excluded(self):
        """Test that non-numeric target is excluded from non-numeric warning."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': ['a', 'b', 'c'],
            'target': ['x', 'y', 'z']  # Non-numeric target
        })
        
        result = validate_data(df, 'target')
        assert "Non-numeric columns detected" in result
        assert "feature2" in result
        assert "target" not in result.split("(may need encoding): ")[1]

    def test_validate_data_mixed_scenarios(self):
        """Test validation with mixed data quality issues."""
        df = pd.DataFrame({
            'feature1': [1, 2, None, None, None],  # >50% null
            'feature2': ['a', 'b', 'c', 'd', 'e'],  # Non-numeric
            'target': [100, 200, 300, 400, 500]
        })
        
        result = validate_data(df, 'target')
        # Should return the first error encountered (null values)
        assert "Columns with >50% null values" in result

    def test_validate_data_with_different_dtypes(self):
        """Test validation with different pandas dtypes."""
        df = pd.DataFrame({
            'int_feature': [1, 2, 3],
            'float_feature': [1.1, 2.2, 3.3],
            'bool_feature': [True, False, True],
            'target': [10, 20, 30]
        })
        
        result = validate_data(df, 'target')
        # Boolean columns are considered non-numeric by pandas select_dtypes
        assert "Non-numeric columns detected" in result
        assert "bool_feature" in result

    def test_validate_data_target_column_case_sensitivity(self):
        """Test that target column name is case sensitive."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'Target': [100, 200, 300]  # Note capital T
        })
        
        result = validate_data(df, 'target')  # lowercase
        assert result == "Target column 'target' not found in data"


class TestFormatModelInfo:
    """Test the format_model_info function."""

    def test_format_model_info_regression(self):
        """Test formatting model info for regression model."""
        # Create a regression model
        model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.9,
            objective='reg:squarederror'
        )
        
        result = format_model_info("test_model", model)
        
        assert "Model: test_model" in result
        assert "Type: Regression" in result
        assert "Status: Trained" in result
        assert "Number of estimators: 50" in result
        assert "Max depth: 4" in result
        assert "Learning rate: 0.1" in result
        assert "Subsample: 0.8" in result
        assert "Column sample by tree: 0.9" in result
        assert "Objective: reg:squarederror" in result
        assert "Model is ready for predictions and analysis" in result

    def test_format_model_info_classification(self):
        """Test formatting model info for classification model."""
        # Create a classification model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.3,
            subsample=1.0,
            colsample_bytree=1.0,
            objective='binary:logistic'
        )
        
        result = format_model_info("classifier_model", model)
        
        assert "Model: classifier_model" in result
        assert "Type: Classification" in result
        assert "Status: Trained" in result
        assert "Number of estimators: 100" in result
        assert "Max depth: 6" in result
        assert "Learning rate: 0.3" in result
        assert "Subsample: 1.0" in result
        assert "Column sample by tree: 1.0" in result
        assert "Objective: binary:logistic" in result

    def test_format_model_info_model_type_detection(self):
        """Test model type detection logic."""
        # Test with regression model (no predict_proba)
        reg_model = xgb.XGBRegressor()
        result = format_model_info("reg", reg_model)
        assert "Type: Regression" in result
        
        # Test with classification model (has predict_proba)
        clf_model = xgb.XGBClassifier()
        result = format_model_info("clf", clf_model)
        assert "Type: Classification" in result

    def test_format_model_info_missing_parameters(self):
        """Test formatting when some parameters are missing."""
        # Create a minimal model
        model = xgb.XGBRegressor()
        
        # Mock get_params to return partial parameters
        with patch.object(model, 'get_params', return_value={'n_estimators': 100}):
            result = format_model_info("minimal_model", model)
            
            assert "Model: minimal_model" in result
            assert "Number of estimators: 100" in result
            assert "Max depth: Not set" in result
            assert "Learning rate: Not set" in result

    def test_format_model_info_custom_model_name(self):
        """Test formatting with various model names."""
        model = xgb.XGBRegressor()
        
        # Test with different model names
        names = ["model_123", "my-model", "test_model_v2"]
        for name in names:
            result = format_model_info(name, model)
            assert f"Model: {name}" in result

    def test_format_model_info_output_structure(self):
        """Test that output has consistent structure."""
        model = xgb.XGBRegressor()
        result = format_model_info("test", model)
        
        lines = result.split('\n')
        assert len(lines) >= 10  # Should have multiple lines
        
        # Check for key sections
        assert any("Model:" in line for line in lines)
        assert any("Type:" in line for line in lines)
        assert any("Status:" in line for line in lines)
        assert any("Key Parameters:" in line for line in lines)
        assert any("Model is ready" in line for line in lines)


class TestFormatFeatureImportance:
    """Test the format_feature_importance function."""

    def test_format_feature_importance_basic(self):
        """Test basic feature importance formatting."""
        # Create a model with feature importance
        model = Mock()
        model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        model.feature_names_in_ = ['feature1', 'feature2', 'feature3']
        
        result = format_feature_importance(model)
        
        assert "Feature Importance (sorted by importance):" in result
        assert "1. feature1: 0.5000" in result
        assert "2. feature2: 0.3000" in result
        assert "3. feature3: 0.2000" in result

    def test_format_feature_importance_no_feature_names(self):
        """Test formatting when feature names are not available."""
        model = Mock()
        model.feature_importances_ = np.array([0.4, 0.6])
        # No feature_names_in_ attribute
        delattr(model, 'feature_names_in_')
        
        result = format_feature_importance(model)
        
        assert "Feature Importance (sorted by importance):" in result
        assert "1. Feature_1: 0.6000" in result
        assert "2. Feature_0: 0.4000" in result

    def test_format_feature_importance_many_features(self):
        """Test formatting with more than 20 features."""
        model = Mock()
        # Create 25 features with decreasing importance
        importances = np.array([1.0 - i * 0.03 for i in range(25)])
        model.feature_importances_ = importances
        model.feature_names_in_ = [f'feature_{i}' for i in range(25)]
        
        result = format_feature_importance(model)
        
        assert "Feature Importance (sorted by importance):" in result
        assert "20. feature_19:" in result
        assert "... and 5 more features" in result
        assert "21. feature_20:" not in result

    def test_format_feature_importance_sorting(self):
        """Test that features are sorted by importance."""
        model = Mock()
        model.feature_importances_ = np.array([0.1, 0.5, 0.3, 0.7])
        model.feature_names_in_ = ['low', 'medium', 'medium_low', 'high']
        
        result = format_feature_importance(model)
        
        lines = result.split('\n')
        # Should be sorted: high (0.7), medium (0.5), medium_low (0.3), low (0.1)
        assert "1. high: 0.7000" in lines[1]
        assert "2. medium: 0.5000" in lines[2]
        assert "3. medium_low: 0.3000" in lines[3]
        assert "4. low: 0.1000" in lines[4]

    def test_format_feature_importance_attribute_error(self):
        """Test error handling when feature_importances_ is not available."""
        model = Mock()
        # Remove feature_importances_ attribute
        delattr(model, 'feature_importances_')
        
        result = format_feature_importance(model)
        
        assert "Error getting feature importance:" in result
        assert "feature_importances_" in result

    def test_format_feature_importance_value_error(self):
        """Test error handling when feature importance has invalid values."""
        # Create a mock that raises an exception when trying to access feature_importances_
        class MockModel:
            @property
            def feature_importances_(self):
                raise ValueError("Invalid feature importance data")
        
        model = MockModel()
        
        result = format_feature_importance(model)
        
        assert "Error getting feature importance:" in result

    def test_format_feature_importance_empty_array(self):
        """Test formatting with empty feature importance array."""
        model = Mock()
        model.feature_importances_ = np.array([])
        model.feature_names_in_ = []
        
        result = format_feature_importance(model)
        
        assert "Feature Importance (sorted by importance):" in result
        # Should handle empty array gracefully

    def test_format_feature_importance_real_xgboost_model(self):
        """Test with a real XGBoost model."""
        # Create sample data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'feature3': [1, 1, 1, 1, 1]
        })
        y = [1, 2, 3, 4, 5]
        
        # Train model
        model = xgb.XGBRegressor(random_state=42)
        model.fit(X, y)
        
        result = format_feature_importance(model)
        
        assert "Feature Importance (sorted by importance):" in result
        assert any("feature" in line for line in result.split('\n'))


class TestPrepareSampleData:
    """Test the prepare_sample_data function."""

    def test_prepare_sample_data_structure(self):
        """Test that sample data has correct structure."""
        data = prepare_sample_data()
        
        assert isinstance(data, dict)
        assert set(data.keys()) == {'feature1', 'feature2', 'feature3', 'target'}

    def test_prepare_sample_data_sizes(self):
        """Test that all data arrays have the same size."""
        data = prepare_sample_data()
        
        sizes = [len(data[key]) for key in data.keys()]
        assert all(size == 100 for size in sizes)

    def test_prepare_sample_data_types(self):
        """Test that all data values are numeric."""
        data = prepare_sample_data()
        
        for key, values in data.items():
            assert isinstance(values, list)
            assert all(isinstance(v, (int, float)) for v in values)

    def test_prepare_sample_data_reproducibility(self):
        """Test that sample data is reproducible."""
        data1 = prepare_sample_data()
        data2 = prepare_sample_data()
        
        # Should be identical due to fixed random seed
        for key in data1.keys():
            assert data1[key] == data2[key]

    def test_prepare_sample_data_target_relationship(self):
        """Test that target has expected relationship to features."""
        data = prepare_sample_data()
        
        # Target should be: 2 * feature1 + 1.5 * feature2 - 0.5 * feature3 + noise
        for i in range(len(data['target'])):
            expected = (2 * data['feature1'][i] + 
                       1.5 * data['feature2'][i] - 
                       0.5 * data['feature3'][i])
            
            # Should be close but not exact due to noise
            assert abs(data['target'][i] - expected) < 1.0

    def test_prepare_sample_data_statistical_properties(self):
        """Test statistical properties of sample data."""
        data = prepare_sample_data()
        
        # Features should have approximately normal distribution
        for feature in ['feature1', 'feature2', 'feature3']:
            values = data[feature]
            mean = np.mean(values)
            std = np.std(values)
            
            # Should be approximately standard normal
            assert abs(mean) < 0.5  # Mean close to 0
            assert abs(std - 1.0) < 0.5  # Std close to 1

    def test_prepare_sample_data_pandas_compatibility(self):
        """Test that sample data works with pandas DataFrame."""
        data = prepare_sample_data()
        
        # Should be able to create DataFrame
        df = pd.DataFrame(data)
        assert df.shape == (100, 4)
        assert list(df.columns) == ['feature1', 'feature2', 'feature3', 'target']

    @patch('src.utils.np.random.seed')
    @patch('src.utils.np.random.randn')
    def test_prepare_sample_data_random_seed_usage(self, mock_randn, mock_seed):
        """Test that random seed is set correctly."""
        mock_randn.return_value = np.array([1, 2, 3])
        
        prepare_sample_data()
        
        mock_seed.assert_called_once_with(42)
        assert mock_randn.call_count >= 4  # Called for each feature and target

    def test_prepare_sample_data_no_nan_values(self):
        """Test that sample data contains no NaN values."""
        data = prepare_sample_data()
        
        for key, values in data.items():
            assert not any(np.isnan(v) for v in values)

    def test_prepare_sample_data_reasonable_ranges(self):
        """Test that sample data values are in reasonable ranges."""
        data = prepare_sample_data()
        
        # All values should be within reasonable bounds
        for key, values in data.items():
            assert all(abs(v) < 10 for v in values)  # Not too extreme


class TestUtilsIntegration:
    """Test integration between utility functions."""

    def test_validate_data_with_sample_data(self):
        """Test validate_data with prepare_sample_data output."""
        sample_data = prepare_sample_data()
        df = pd.DataFrame(sample_data)
        
        result = validate_data(df, 'target')
        assert result == "valid"

    def test_utils_module_imports(self):
        """Test that all utility functions can be imported."""
        from src.utils import (
            validate_data as imported_validate,
            format_model_info as imported_format_info,
            format_feature_importance as imported_format_importance,
            prepare_sample_data as imported_prepare_data,
        )
        
        # Should be the same functions
        assert imported_validate is validate_data
        assert imported_format_info is format_model_info
        assert imported_format_importance is format_feature_importance
        assert imported_prepare_data is prepare_sample_data

    def test_utils_with_real_xgboost_workflow(self):
        """Test utilities in a complete XGBoost workflow."""
        # Generate sample data
        sample_data = prepare_sample_data()
        df = pd.DataFrame(sample_data)
        
        # Validate data
        validation_result = validate_data(df, 'target')
        assert validation_result == "valid"
        
        # Train model
        X = df.drop('target', axis=1)
        y = df['target']
        model = xgb.XGBRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Format model info
        model_info = format_model_info("test_model", model)
        assert "Model: test_model" in model_info
        assert "Type: Regression" in model_info
        
        # Format feature importance
        feature_importance = format_feature_importance(model)
        assert "Feature Importance" in feature_importance
        assert "feature1" in feature_importance or "Feature_0" in feature_importance 