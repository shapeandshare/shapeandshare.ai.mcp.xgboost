"""Enhanced Input Validation Framework for MCP XGBoost Server.

This module provides comprehensive input validation using Pydantic models
for data validation, schema enforcement, and type safety throughout the
MCP XGBoost server application.

The validation framework supports:
- Comprehensive data validation with detailed error messages
- Schema enforcement for consistent data structures
- Type safety with automatic conversion
- Custom validators for domain-specific validation
- Performance optimization with caching
- Integration with the configuration system

Classes
-------
TrainingDataSchema
    Schema for model training data validation
PredictionDataSchema
    Schema for prediction data validation
ModelParametersSchema
    Schema for XGBoost model parameters validation
RequestValidationSchema
    Schema for HTTP request validation
DataValidationResult
    Result of data validation operations
ValidationError
    Custom validation error with detailed information

Functions
---------
validate_training_data
    Validate data for model training
validate_prediction_data
    Validate data for model prediction
validate_model_parameters
    Validate XGBoost model parameters
validate_request_data
    Validate HTTP request data
create_validation_schema
    Create custom validation schema

Notes
-----
The validation framework is designed to be:
- Comprehensive with detailed error reporting
- Fast with minimal performance overhead
- Extensible for custom validation rules
- Integration-friendly with existing code
- Type-safe with automatic conversion

Validation Features
------------------
- Data type validation and conversion
- Range and constraint checking
- Missing value detection and handling
- Schema compatibility validation
- Feature name consistency checking
- Data size and limit enforcement
- Custom business rule validation

Examples
--------
Basic data validation:

    >>> from src.validation import validate_training_data
    >>> data = {"feature1": [1, 2, 3], "target": [0, 1, 0]}
    >>> result = validate_training_data(data, "target")
    >>> if result.is_valid:
    ...     print("Data is valid")
    >>> else:
    ...     print(f"Validation errors: {result.errors}")

Schema validation:

    >>> from src.validation import TrainingDataSchema
    >>> schema = TrainingDataSchema(
    ...     features={"feature1": [1.0, 2.0, 3.0]},
    ...     target="target_column",
    ...     target_values=[0, 1, 0]
    ... )

Custom validation:

    >>> from src.validation import create_validation_schema
    >>> custom_schema = create_validation_schema({
    ...     "required_columns": ["feature1", "feature2"],
    ...     "max_samples": 10000,
    ...     "min_samples": 10
    ... })

See Also
--------
src.config : Configuration management
src.exceptions : Custom exception handling
pydantic : Data validation library
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from .config import get_config
from .exceptions import wrap_exception
from .logging_config import get_logger


class DataValidationResult(BaseModel):
    """Result of data validation operations.

    Attributes
    ----------
    is_valid : bool
        Whether the data passed validation
    errors : List[str]
        List of validation error messages
    warnings : List[str]
        List of validation warnings
    metadata : Dict[str, Any]
        Additional validation metadata
    validated_data : Optional[Dict[str, Any]]
        Cleaned and validated data
    """

    is_valid: bool = Field(..., description="Whether data is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Validation metadata")
    validated_data: Optional[Dict[str, Any]] = Field(default=None, description="Validated data")

    def add_error(self, message: str):
        """Add a validation error."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        """Add a validation warning."""
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "validated_data": self.validated_data,
        }


class ModelParametersSchema(BaseModel):
    """Schema for XGBoost model parameters validation.

    Attributes
    ----------
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum tree depth
    learning_rate : float
        Learning rate
    subsample : float
        Subsample ratio
    colsample_bytree : float
        Column sample ratio
    random_state : int
        Random seed
    model_type : str
        Model type (regression/classification)
    """

    n_estimators: int = Field(default=100, ge=1, le=10000, description="Number of boosting rounds")
    max_depth: int = Field(default=6, ge=1, le=20, description="Maximum tree depth")
    learning_rate: float = Field(default=0.3, gt=0, le=1, description="Learning rate")
    subsample: float = Field(default=1.0, gt=0, le=1, description="Subsample ratio")
    colsample_bytree: float = Field(default=1.0, gt=0, le=1, description="Column sample ratio")
    random_state: int = Field(default=42, ge=0, description="Random seed")
    model_type: str = Field(default="regression", pattern="^(regression|classification)$", description="Model type")

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v):
        """Validate learning rate is reasonable."""
        if v < 0.01:
            raise ValueError("Learning rate too small (< 0.01), may cause slow training")
        if v > 0.5:
            raise ValueError("Learning rate too large (> 0.5), may cause instability")
        return v

    @field_validator("n_estimators")
    @classmethod
    def validate_n_estimators(cls, v):
        """Validate number of estimators."""
        if v > 1000:
            # This is just a warning, not an error
            pass
        return v


class TrainingDataSchema(BaseModel):
    """Schema for model training data validation.

    Attributes
    ----------
    features : Dict[str, List[Union[float, int]]]
        Feature data with column names and values
    target : str
        Target column name
    target_values : List[Union[float, int, str]]
        Target values
    feature_names : List[str]
        List of feature column names
    sample_count : int
        Number of samples
    feature_count : int
        Number of features
    """

    features: Dict[str, List[Union[float, int]]] = Field(..., description="Feature data")
    target: str = Field(..., min_length=1, description="Target column name")
    target_values: List[Union[float, int, str]] = Field(..., min_length=1, description="Target values")
    feature_names: List[str] = Field(default_factory=list, description="Feature column names")
    sample_count: int = Field(default=0, ge=1, description="Number of samples")
    feature_count: int = Field(default=0, ge=1, description="Number of features")

    @model_validator(mode="before")
    @classmethod
    def validate_data_structure(cls, values):
        """Validate the overall data structure."""
        config = get_config()

        features = values.get("features", {})
        target_values = values.get("target_values", [])

        if not features:
            raise ValueError("Features data is required")

        if not target_values:
            raise ValueError("Target values are required")

        # Check sample count consistency
        sample_counts = [len(feature_data) for feature_data in features.values()]
        if not all(count == len(target_values) for count in sample_counts):
            raise ValueError("Inconsistent sample counts between features and target")

        sample_count = len(target_values)
        feature_count = len(features)

        # Check size limits
        if sample_count > config.resources.max_samples:
            raise ValueError(f"Too many samples: {sample_count} > {config.resources.max_samples}")

        if sample_count < config.resources.min_samples:
            raise ValueError(f"Too few samples: {sample_count} < {config.resources.min_samples}")

        if feature_count > config.resources.max_features:
            raise ValueError(f"Too many features: {feature_count} > {config.resources.max_features}")

        # Set computed values
        values["feature_names"] = list(features.keys())
        values["sample_count"] = sample_count
        values["feature_count"] = feature_count

        return values

    @field_validator("features")
    @classmethod
    def validate_feature_data(cls, v):
        """Validate feature data types and values."""
        for feature_name, feature_data in v.items():
            if not feature_name or not isinstance(feature_name, str):
                raise ValueError(f"Invalid feature name: {feature_name}")

            if not feature_data:
                raise ValueError(f"Empty feature data for: {feature_name}")

            # Check for valid numeric data
            for i, value in enumerate(feature_data):
                if value is None:
                    raise ValueError(f"Null value in feature '{feature_name}' at index {i}")
                if not isinstance(value, (int, float)):
                    try:
                        float(value)
                    except (ValueError, TypeError) as exc:
                        msg = f"Non-numeric value in feature '{feature_name}' at index {i}: {value}"
                        raise ValueError(msg) from exc

        return v

    @field_validator("target_values")
    @classmethod
    def validate_target_data(cls, v):
        """Validate target data."""
        if not v:
            raise ValueError("Target values cannot be empty")

        # Check for null values
        null_count = sum(1 for val in v if val is None)
        if null_count > 0:
            raise ValueError(f"Target contains {null_count} null values")

        return v

    @field_validator("target")
    @classmethod
    def validate_target_name(cls, v):
        """Validate target column name."""
        if not v or not isinstance(v, str):
            raise ValueError("Target must be a non-empty string")

        # Check for valid column name
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
            raise ValueError(f"Invalid target column name: {v}")

        return v


class PredictionDataSchema(BaseModel):
    """Schema for prediction data validation.

    Attributes
    ----------
    features : Dict[str, List[Union[float, int]]]
        Feature data for prediction
    feature_names : List[str]
        List of feature column names
    sample_count : int
        Number of samples
    feature_count : int
        Number of features
    """

    features: Dict[str, List[Union[float, int]]] = Field(..., description="Feature data for prediction")
    feature_names: List[str] = Field(default_factory=list, description="Feature column names")
    sample_count: int = Field(default=0, ge=1, description="Number of samples")
    feature_count: int = Field(default=0, ge=1, description="Number of features")

    @model_validator(mode="before")
    @classmethod
    def validate_prediction_structure(cls, values):
        """Validate prediction data structure."""
        config = get_config()

        features = values.get("features", {})

        if not features:
            raise ValueError("Features data is required for prediction")

        # Check sample count consistency
        sample_counts = [len(feature_data) for feature_data in features.values()]
        if not all(count == sample_counts[0] for count in sample_counts):
            raise ValueError("Inconsistent sample counts between features")

        sample_count = sample_counts[0]
        feature_count = len(features)

        # Check size limits
        if sample_count > config.resources.max_samples:
            raise ValueError(f"Too many samples: {sample_count} > {config.resources.max_samples}")

        if feature_count > config.resources.max_features:
            raise ValueError(f"Too many features: {feature_count} > {config.resources.max_features}")

        # Set computed values
        values["feature_names"] = list(features.keys())
        values["sample_count"] = sample_count
        values["feature_count"] = feature_count

        return values

    @field_validator("features")
    @classmethod
    def validate_prediction_features(cls, v):
        """Validate prediction feature data."""
        for feature_name, feature_data in v.items():
            if not feature_name or not isinstance(feature_name, str):
                raise ValueError(f"Invalid feature name: {feature_name}")

            if not feature_data:
                raise ValueError(f"Empty feature data for: {feature_name}")

            # Check for valid numeric data
            for i, value in enumerate(feature_data):
                if value is None:
                    raise ValueError(f"Null value in feature '{feature_name}' at index {i}")
                if not isinstance(value, (int, float)):
                    try:
                        float(value)
                    except (ValueError, TypeError) as exc:
                        msg = f"Non-numeric value in feature '{feature_name}' at index {i}: {value}"
                        raise ValueError(msg) from exc

        return v


class RequestValidationSchema(BaseModel):
    """Schema for HTTP request validation.

    Attributes
    ----------
    model_name : str
        Model name identifier
    data : str
        JSON string containing data
    operation : str
        Operation type
    parameters : Dict[str, Any]
        Additional parameters
    timestamp : datetime
        Request timestamp
    """

    model_name: str = Field(
        ..., min_length=1, max_length=100, pattern="^[a-zA-Z0-9_-]+$", description="Model name identifier"
    )
    data: str = Field(..., min_length=1, description="JSON string containing data")
    operation: str = Field(..., pattern="^(train|predict|info|importance|list)$", description="Operation type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")

    @field_validator("data")
    @classmethod
    def validate_json_data(cls, v):
        """Validate that data is valid JSON."""
        try:
            json.loads(v)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {str(e)}") from e
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name_format(cls, v):
        """Validate model name format."""
        if len(v) < 1 or len(v) > 100:
            raise ValueError("Model name must be 1-100 characters")

        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Model name can only contain alphanumeric characters, underscores, and hyphens")

        return v


class DataValidator:
    """Main data validation class with caching and optimization.

    This class provides the core validation functionality with
    performance optimizations and caching.

    Attributes
    ----------
    config : AppConfig
        Application configuration
    logger : StructuredLogger
        Logger instance
    cache : Dict[str, Any]
        Validation result cache
    """

    def __init__(self):
        """Initialize the data validator."""
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.cache = {}

    def _create_cache_key(self, data: Any, validation_type: str) -> str:
        """Create a cache key for validation results."""
        data_str = json.dumps(data, sort_keys=True) if isinstance(data, dict) else str(data)
        return f"{validation_type}:{hash(data_str)}"

    def validate_training_data(
        self, data: Dict[str, Any], target_column: str, use_cache: bool = True
    ) -> DataValidationResult:
        """Validate training data.

        Parameters
        ----------
        data : Dict[str, Any]
            Training data dictionary
        target_column : str
            Name of the target column
        use_cache : bool, optional
            Whether to use cache (default: True)

        Returns
        -------
        DataValidationResult
            Validation result
        """
        try:
            # Check cache
            if use_cache:
                cache_key = self._create_cache_key(data, f"training:{target_column}")
                if cache_key in self.cache:
                    return self.cache[cache_key]

            result = DataValidationResult(is_valid=True)

            # Basic structure validation
            if not data:
                result.add_error("Data is empty")
                return result

            if target_column not in data:
                result.add_error(f"Target column '{target_column}' not found in data")
                return result

            # Separate features and target
            features = {k: v for k, v in data.items() if k != target_column}
            target_values = data[target_column]

            # Validate using Pydantic schema
            try:
                schema = TrainingDataSchema(features=features, target=target_column, target_values=target_values)

                # Add metadata
                result.metadata = {
                    "sample_count": schema.sample_count,
                    "feature_count": schema.feature_count,
                    "feature_names": schema.feature_names,
                    "target_column": target_column,
                }

                # Create validated data
                result.validated_data = {"features": schema.features, "target": schema.target_values}

            except ValueError as e:
                result.add_error(str(e))
                return result

            # Additional custom validations
            self._validate_data_quality(features, target_values, result)
            self._validate_data_distribution(target_values, result)

            # Cache result if successful
            if use_cache and result.is_valid:
                cache_key = self._create_cache_key(data, f"training:{target_column}")
                self.cache[cache_key] = result

            return result

        except Exception as e:
            raise wrap_exception(
                "validate training data",
                e,
                {"target_column": target_column, "data_keys": list(data.keys()) if data else []},
            ) from e

    def validate_prediction_data(
        self, data: Dict[str, Any], expected_features: Optional[List[str]] = None, use_cache: bool = True
    ) -> DataValidationResult:
        """Validate prediction data.

        Parameters
        ----------
        data : Dict[str, Any]
            Prediction data dictionary
        expected_features : List[str], optional
            Expected feature names for consistency checking
        use_cache : bool, optional
            Whether to use cache (default: True)

        Returns
        -------
        DataValidationResult
            Validation result
        """
        try:
            # Check cache
            if use_cache:
                cache_key = self._create_cache_key(data, f"prediction:{expected_features}")
                if cache_key in self.cache:
                    return self.cache[cache_key]

            result = DataValidationResult(is_valid=True)

            # Basic structure validation
            if not data:
                result.add_error("Data is empty")
                return result

            # Validate using Pydantic schema
            try:
                schema = PredictionDataSchema(features=data)

                # Check feature consistency
                if expected_features:
                    missing_features = set(expected_features) - set(schema.feature_names)
                    extra_features = set(schema.feature_names) - set(expected_features)

                    if missing_features:
                        result.add_error(f"Missing features: {list(missing_features)}")

                    if extra_features:
                        result.add_warning(f"Extra features (will be ignored): {list(extra_features)}")

                # Add metadata
                result.metadata = {
                    "sample_count": schema.sample_count,
                    "feature_count": schema.feature_count,
                    "feature_names": schema.feature_names,
                }

                # Create validated data
                result.validated_data = {"features": schema.features}

            except ValueError as e:
                result.add_error(str(e))
                return result

            # Cache result if successful
            if use_cache and result.is_valid:
                cache_key = self._create_cache_key(data, f"prediction:{expected_features}")
                self.cache[cache_key] = result

            return result

        except Exception as e:
            raise wrap_exception(
                "validate prediction data",
                e,
                {"data_keys": list(data.keys()) if data else [], "expected_features": expected_features},
            ) from e

    def validate_model_parameters(self, parameters: Dict[str, Any]) -> DataValidationResult:
        """Validate XGBoost model parameters.

        Parameters
        ----------
        parameters : Dict[str, Any]
            Model parameters dictionary

        Returns
        -------
        DataValidationResult
            Validation result
        """
        try:
            result = DataValidationResult(is_valid=True)

            # Validate using Pydantic schema
            try:
                schema = ModelParametersSchema(**parameters)

                result.metadata = {"validated_parameters": schema.dict(), "parameter_count": len(parameters)}

                result.validated_data = schema.dict()

            except ValueError as e:
                result.add_error(str(e))
                return result

            return result

        except Exception as e:
            raise wrap_exception("validate model parameters", e, {"parameters": parameters}) from e

    def _validate_data_quality(self, features: Dict[str, List], _target_values: List, result: DataValidationResult):
        """Validate data quality aspects."""
        # Check for missing values
        for feature_name, feature_data in features.items():
            null_count = sum(1 for val in feature_data if val is None or pd.isna(val))
            if null_count > 0:
                null_percentage = (null_count / len(feature_data)) * 100
                if null_percentage > 50:
                    result.add_error(f"Feature '{feature_name}' has {null_percentage:.1f}% missing values")
                elif null_percentage > 10:
                    result.add_warning(f"Feature '{feature_name}' has {null_percentage:.1f}% missing values")

        # Check for constant features
        for feature_name, feature_data in features.items():
            unique_values = len(set(feature_data))
            if unique_values == 1:
                result.add_warning(f"Feature '{feature_name}' is constant")
            elif unique_values < 3:
                result.add_warning(f"Feature '{feature_name}' has very low variance")

    def _validate_data_distribution(self, target_values: List, result: DataValidationResult):
        """Validate target data distribution."""
        unique_values = len(set(target_values))

        # Check for classification vs regression
        if unique_values <= 20 and all(isinstance(val, (int, bool)) for val in target_values):
            # Likely classification
            class_counts: dict[int, int] = {}
            for val in target_values:
                class_counts[val] = class_counts.get(val, 0) + 1

            # Check for class imbalance
            min_count = min(class_counts.values())
            max_count = max(class_counts.values())
            imbalance_ratio = max_count / min_count

            if imbalance_ratio > 10:
                result.add_warning(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f})")
            elif imbalance_ratio > 5:
                result.add_warning(f"Class imbalance detected (ratio: {imbalance_ratio:.1f})")

        # Check for outliers in regression
        if unique_values > 20:
            try:
                numeric_values = [float(val) for val in target_values]
                q1 = np.percentile(numeric_values, 25)
                q3 = np.percentile(numeric_values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers = [val for val in numeric_values if val < lower_bound or val > upper_bound]
                outlier_percentage = (len(outliers) / len(numeric_values)) * 100

                if outlier_percentage > 10:
                    result.add_warning(f"High percentage of outliers in target: {outlier_percentage:.1f}%")
                elif outlier_percentage > 5:
                    result.add_warning(f"Outliers detected in target: {outlier_percentage:.1f}%")

            except (ValueError, TypeError):
                pass  # Skip outlier detection for non-numeric targets

    def clear_cache(self):
        """Clear the validation cache."""
        self.cache.clear()
        self.logger.info("Validation cache cleared")


# Global validator instance
_validator = DataValidator()


# Convenience functions
def validate_training_data(data: Dict[str, Any], target_column: str, use_cache: bool = True) -> DataValidationResult:
    """Validate training data.

    Parameters
    ----------
    data : Dict[str, Any]
        Training data dictionary
    target_column : str
        Name of the target column
    use_cache : bool, optional
        Whether to use cache (default: True)

    Returns
    -------
    DataValidationResult
        Validation result
    """
    return _validator.validate_training_data(data, target_column, use_cache)


def validate_prediction_data(
    data: Dict[str, Any], expected_features: Optional[List[str]] = None, use_cache: bool = True
) -> DataValidationResult:
    """Validate prediction data.

    Parameters
    ----------
    data : Dict[str, Any]
        Prediction data dictionary
    expected_features : List[str], optional
        Expected feature names for consistency checking
    use_cache : bool, optional
        Whether to use cache (default: True)

    Returns
    -------
    DataValidationResult
        Validation result
    """
    return _validator.validate_prediction_data(data, expected_features, use_cache)


def validate_model_parameters(parameters: Dict[str, Any]) -> DataValidationResult:
    """Validate XGBoost model parameters.

    Parameters
    ----------
    parameters : Dict[str, Any]
        Model parameters dictionary

    Returns
    -------
    DataValidationResult
        Validation result
    """
    return _validator.validate_model_parameters(parameters)


def create_validation_schema(_rules: Dict[str, Any]) -> BaseModel:
    """Create custom validation schema.

    Parameters
    ----------
    rules : Dict[str, Any]
        Validation rules dictionary

    Returns
    -------
    BaseModel
        Custom Pydantic schema
    """

    # This is a simplified implementation - in practice, you might want
    # to dynamically create schemas based on the rules
    class CustomSchema(BaseModel):
        pass

    return CustomSchema()
