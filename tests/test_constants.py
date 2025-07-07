"""Tests for src/constants.py module.

This module tests all constants and configuration values defined in the
constants module to ensure they are properly set and accessible.
"""

import pytest
from src.constants import (
    APP_NAME,
    USER_AGENT,
    DEFAULT_XGBOOST_PARAMS,
    MAX_FEATURES,
    MAX_SAMPLES,
    MIN_SAMPLES,
    SUPPORTED_DATA_FORMATS,
    MAX_FEATURE_IMPORTANCE_DISPLAY,
)


class TestApplicationConstants:
    """Test application identification constants."""

    def test_app_name_constant(self):
        """Test APP_NAME constant is properly set."""
        assert APP_NAME == "mcp-xgboost"
        assert isinstance(APP_NAME, str)
        assert len(APP_NAME) > 0

    def test_user_agent_constant(self):
        """Test USER_AGENT constant is properly set."""
        assert USER_AGENT == "mcp-xgboost/1.0"
        assert isinstance(USER_AGENT, str)
        assert APP_NAME in USER_AGENT
        assert "/" in USER_AGENT


class TestXGBoostConstants:
    """Test XGBoost-related constants."""

    def test_default_xgboost_params_structure(self):
        """Test DEFAULT_XGBOOST_PARAMS has correct structure."""
        assert isinstance(DEFAULT_XGBOOST_PARAMS, dict)
        assert len(DEFAULT_XGBOOST_PARAMS) > 0

    def test_default_xgboost_params_keys(self):
        """Test DEFAULT_XGBOOST_PARAMS has expected keys."""
        expected_keys = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "random_state",
        }
        assert set(DEFAULT_XGBOOST_PARAMS.keys()) == expected_keys

    def test_default_xgboost_params_values(self):
        """Test DEFAULT_XGBOOST_PARAMS has expected values."""
        assert DEFAULT_XGBOOST_PARAMS["n_estimators"] == 100
        assert DEFAULT_XGBOOST_PARAMS["max_depth"] == 6
        assert DEFAULT_XGBOOST_PARAMS["learning_rate"] == 0.3
        assert DEFAULT_XGBOOST_PARAMS["subsample"] == 1.0
        assert DEFAULT_XGBOOST_PARAMS["colsample_bytree"] == 1.0
        assert DEFAULT_XGBOOST_PARAMS["random_state"] == 42

    def test_default_xgboost_params_types(self):
        """Test DEFAULT_XGBOOST_PARAMS has correct value types."""
        assert isinstance(DEFAULT_XGBOOST_PARAMS["n_estimators"], int)
        assert isinstance(DEFAULT_XGBOOST_PARAMS["max_depth"], int)
        assert isinstance(DEFAULT_XGBOOST_PARAMS["learning_rate"], float)
        assert isinstance(DEFAULT_XGBOOST_PARAMS["subsample"], float)
        assert isinstance(DEFAULT_XGBOOST_PARAMS["colsample_bytree"], float)
        assert isinstance(DEFAULT_XGBOOST_PARAMS["random_state"], int)

    def test_default_xgboost_params_valid_ranges(self):
        """Test DEFAULT_XGBOOST_PARAMS values are in valid ranges."""
        assert DEFAULT_XGBOOST_PARAMS["n_estimators"] > 0
        assert DEFAULT_XGBOOST_PARAMS["max_depth"] > 0
        assert 0 < DEFAULT_XGBOOST_PARAMS["learning_rate"] <= 1
        assert 0 < DEFAULT_XGBOOST_PARAMS["subsample"] <= 1
        assert 0 < DEFAULT_XGBOOST_PARAMS["colsample_bytree"] <= 1
        assert DEFAULT_XGBOOST_PARAMS["random_state"] >= 0


class TestValidationConstants:
    """Test data validation constants."""

    def test_max_features_constant(self):
        """Test MAX_FEATURES constant is properly set."""
        assert MAX_FEATURES == 1000
        assert isinstance(MAX_FEATURES, int)
        assert MAX_FEATURES > 0

    def test_max_samples_constant(self):
        """Test MAX_SAMPLES constant is properly set."""
        assert MAX_SAMPLES == 100000
        assert isinstance(MAX_SAMPLES, int)
        assert MAX_SAMPLES > 0

    def test_min_samples_constant(self):
        """Test MIN_SAMPLES constant is properly set."""
        assert MIN_SAMPLES == 10
        assert isinstance(MIN_SAMPLES, int)
        assert MIN_SAMPLES > 0

    def test_validation_limits_relationship(self):
        """Test validation limits have logical relationship."""
        assert MIN_SAMPLES < MAX_SAMPLES
        assert MAX_FEATURES > 0
        assert MIN_SAMPLES >= 1


class TestDataFormatConstants:
    """Test supported data format constants."""

    def test_supported_data_formats_structure(self):
        """Test SUPPORTED_DATA_FORMATS has correct structure."""
        assert isinstance(SUPPORTED_DATA_FORMATS, list)
        assert len(SUPPORTED_DATA_FORMATS) > 0

    def test_supported_data_formats_values(self):
        """Test SUPPORTED_DATA_FORMATS has expected values."""
        expected_formats = ["json", "csv"]
        assert SUPPORTED_DATA_FORMATS == expected_formats

    def test_supported_data_formats_types(self):
        """Test SUPPORTED_DATA_FORMATS contains strings."""
        for format_type in SUPPORTED_DATA_FORMATS:
            assert isinstance(format_type, str)
            assert len(format_type) > 0


class TestDisplayConstants:
    """Test display-related constants."""

    def test_max_feature_importance_display_constant(self):
        """Test MAX_FEATURE_IMPORTANCE_DISPLAY constant is properly set."""
        assert MAX_FEATURE_IMPORTANCE_DISPLAY == 20
        assert isinstance(MAX_FEATURE_IMPORTANCE_DISPLAY, int)
        assert MAX_FEATURE_IMPORTANCE_DISPLAY > 0

    def test_max_feature_importance_display_reasonable(self):
        """Test MAX_FEATURE_IMPORTANCE_DISPLAY is reasonable for display."""
        assert MAX_FEATURE_IMPORTANCE_DISPLAY <= 100  # Not too many to display
        assert MAX_FEATURE_IMPORTANCE_DISPLAY >= 5    # Not too few to be useful


class TestConstantsIntegration:
    """Test integration between constants."""

    def test_app_name_in_user_agent(self):
        """Test APP_NAME is properly integrated in USER_AGENT."""
        assert APP_NAME in USER_AGENT

    def test_constants_module_imports(self):
        """Test all constants can be imported from module."""
        from src.constants import (
            APP_NAME as imported_app_name,
            USER_AGENT as imported_user_agent,
            DEFAULT_XGBOOST_PARAMS as imported_params,
            MAX_FEATURES as imported_max_features,
            MAX_SAMPLES as imported_max_samples,
            MIN_SAMPLES as imported_min_samples,
            SUPPORTED_DATA_FORMATS as imported_formats,
            MAX_FEATURE_IMPORTANCE_DISPLAY as imported_display,
        )
        
        assert imported_app_name == APP_NAME
        assert imported_user_agent == USER_AGENT
        assert imported_params == DEFAULT_XGBOOST_PARAMS
        assert imported_max_features == MAX_FEATURES
        assert imported_max_samples == MAX_SAMPLES
        assert imported_min_samples == MIN_SAMPLES
        assert imported_formats == SUPPORTED_DATA_FORMATS
        assert imported_display == MAX_FEATURE_IMPORTANCE_DISPLAY

    def test_constants_immutability(self):
        """Test constants maintain their values (immutability check)."""
        # Test that constants are not accidentally modified
        original_app_name = APP_NAME
        original_user_agent = USER_AGENT
        original_params = DEFAULT_XGBOOST_PARAMS.copy()
        
        # Import again to verify values haven't changed
        from src.constants import (
            APP_NAME as check_app_name,
            USER_AGENT as check_user_agent,
            DEFAULT_XGBOOST_PARAMS as check_params,
        )
        
        assert check_app_name == original_app_name
        assert check_user_agent == original_user_agent
        assert check_params == original_params

    def test_all_constants_defined(self):
        """Test all expected constants are defined in module."""
        import src.constants as constants_module
        
        expected_constants = [
            "APP_NAME",
            "USER_AGENT", 
            "DEFAULT_XGBOOST_PARAMS",
            "MAX_FEATURES",
            "MAX_SAMPLES",
            "MIN_SAMPLES",
            "SUPPORTED_DATA_FORMATS",
            "MAX_FEATURE_IMPORTANCE_DISPLAY",
        ]
        
        for constant_name in expected_constants:
            assert hasattr(constants_module, constant_name)
            assert getattr(constants_module, constant_name) is not None 