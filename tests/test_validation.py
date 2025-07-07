"""Tests for the validation module."""

import pytest
import json
from unittest.mock import patch

from src.validation import (
    DataValidationResult,
    ModelParametersSchema,
    TrainingDataSchema,
    PredictionDataSchema,
    RequestValidationSchema,
    DataValidator,
    validate_training_data,
    validate_prediction_data,
    validate_model_parameters,
)


class TestDataValidationResult:
    """Test the DataValidationResult class."""

    def test_valid_result(self):
        """Test creating a valid result."""
        result = DataValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_result_with_errors(self):
        """Test result with errors."""
        result = DataValidationResult(is_valid=False)
        result.add_error("Test error")
        assert result.is_valid is False
        assert "Test error" in result.errors

    def test_result_with_warnings(self):
        """Test result with warnings."""
        result = DataValidationResult(is_valid=True)
        result.add_warning("Test warning")
        assert result.is_valid is True
        assert "Test warning" in result.warnings

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = DataValidationResult(is_valid=True)
        result.add_error("Test error")
        result.add_warning("Test warning")
        
        result_dict = result.to_dict()
        assert result_dict["is_valid"] is False  # Should be False due to error
        assert "Test error" in result_dict["errors"]
        assert "Test warning" in result_dict["warnings"]


class TestModelParametersSchema:
    """Test the ModelParametersSchema class."""

    def test_valid_parameters(self):
        """Test validation of valid parameters."""
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.3,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'random_state': 42,
            'model_type': 'regression'
        }
        
        schema = ModelParametersSchema(**params)
        assert schema.n_estimators == 100
        assert schema.max_depth == 6
        assert schema.learning_rate == 0.3
        assert schema.model_type == 'regression'

    def test_default_parameters(self):
        """Test default parameter values."""
        schema = ModelParametersSchema()
        assert schema.n_estimators == 100
        assert schema.max_depth == 6
        assert schema.learning_rate == 0.3
        assert schema.model_type == 'regression'

    def test_invalid_learning_rate(self):
        """Test validation fails with invalid learning rate."""
        with pytest.raises(ValueError, match="Learning rate too small"):
            ModelParametersSchema(learning_rate=0.005)

        with pytest.raises(ValueError, match="Learning rate too large"):
            ModelParametersSchema(learning_rate=0.6)

    def test_invalid_model_type(self):
        """Test validation fails with invalid model type."""
        with pytest.raises(ValueError):
            ModelParametersSchema(model_type='invalid')


class TestTrainingDataSchema:
    """Test the TrainingDataSchema class."""

    def test_valid_training_data(self):
        """Test validation of valid training data."""
        data = {
            'features': {
                'feature1': [1.0, 2.0, 3.0],
                'feature2': [0.5, 1.5, 2.5]
            },
            'target': 'target_col',
            'target_values': [1, 0, 1]
        }
        
        schema = TrainingDataSchema(**data)
        assert len(schema.features) == 2
        assert schema.target == 'target_col'
        assert len(schema.target_values) == 3

    def test_invalid_empty_features(self):
        """Test validation fails with empty features."""
        data = {
            'features': {},
            'target': 'target_col',
            'target_values': [1, 0, 1]
        }
        
        with pytest.raises(ValueError, match="Features data is required"):
            TrainingDataSchema(**data)

    def test_invalid_empty_target(self):
        """Test validation fails with empty target values."""
        data = {
            'features': {'feature1': [1.0, 2.0, 3.0]},
            'target': 'target_col',
            'target_values': []
        }
        
        with pytest.raises(ValueError, match="Target values are required"):
            TrainingDataSchema(**data)


class TestPredictionDataSchema:
    """Test the PredictionDataSchema class."""

    def test_valid_prediction_data(self):
        """Test validation of valid prediction data."""
        data = {
            'features': {
                'feature1': [1.0, 2.0, 3.0],
                'feature2': [0.5, 1.5, 2.5]
            }
        }
        
        schema = PredictionDataSchema(**data)
        assert len(schema.features) == 2
        assert schema.sample_count == 3
        assert schema.feature_count == 2

    def test_invalid_empty_features(self):
        """Test validation fails with empty features."""
        data = {'features': {}}
        
        with pytest.raises(ValueError, match="Features data is required"):
            PredictionDataSchema(**data)


class TestRequestValidationSchema:
    """Test the RequestValidationSchema class."""

    def test_valid_request(self):
        """Test validation of valid request."""
        data = {
            'model_name': 'test_model',
            'data': '{"feature1": [1, 2, 3]}',
            'operation': 'predict'
        }
        
        schema = RequestValidationSchema(**data)
        assert schema.model_name == 'test_model'
        assert schema.operation == 'predict'

    def test_invalid_model_name(self):
        """Test validation fails with invalid model name."""
        data = {
            'model_name': 'invalid name!',  # Contains space and special char
            'data': '{"feature1": [1, 2, 3]}',
            'operation': 'predict'
        }
        
        with pytest.raises(ValueError):
            RequestValidationSchema(**data)

    def test_invalid_json_data(self):
        """Test validation fails with invalid JSON."""
        data = {
            'model_name': 'test_model',
            'data': 'invalid json',
            'operation': 'predict'
        }
        
        with pytest.raises(ValueError, match="Invalid JSON data"):
            RequestValidationSchema(**data)

    def test_invalid_operation(self):
        """Test validation fails with invalid operation."""
        data = {
            'model_name': 'test_model',
            'data': '{"feature1": [1, 2, 3]}',
            'operation': 'invalid_op'
        }
        
        with pytest.raises(ValueError):
            RequestValidationSchema(**data)


class TestDataValidator:
    """Test the DataValidator class."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = DataValidator()
        assert validator.cache == {}

    def test_validate_training_data_success(self):
        """Test successful training data validation."""
        validator = DataValidator()
        data = {
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [0.5, 1.5, 2.5],
            'target': [1, 0, 1]
        }
        
        result = validator.validate_training_data(data, 'target')
        assert isinstance(result, DataValidationResult)
        assert result.is_valid is True

    def test_validate_prediction_data_success(self):
        """Test successful prediction data validation."""
        validator = DataValidator()
        data = {
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [0.5, 1.5, 2.5]
        }
        
        result = validator.validate_prediction_data(data)
        assert isinstance(result, DataValidationResult)
        assert result.is_valid is True

    def test_validate_model_parameters_success(self):
        """Test successful model parameters validation."""
        validator = DataValidator()
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.3
        }
        
        result = validator.validate_model_parameters(params)
        assert isinstance(result, DataValidationResult)
        assert result.is_valid is True


class TestValidationFunctions:
    """Test validation utility functions."""

    def test_validate_training_data_function(self):
        """Test validate_training_data function."""
        data = {
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [0.5, 1.5, 2.5],
            'target': [1, 0, 1]
        }
        
        result = validate_training_data(data, 'target')
        assert isinstance(result, DataValidationResult)

    def test_validate_prediction_data_function(self):
        """Test validate_prediction_data function."""
        data = {
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [0.5, 1.5, 2.5]
        }
        
        result = validate_prediction_data(data)
        assert isinstance(result, DataValidationResult)

    def test_validate_model_parameters_function(self):
        """Test validate_model_parameters function."""
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.3
        }
        
        result = validate_model_parameters(params)
        assert isinstance(result, DataValidationResult) 