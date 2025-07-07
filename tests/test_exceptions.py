"""Tests for the exceptions module."""

import pytest
from src.exceptions import (
    MCPXGBoostException,
    ConfigurationError,
    DataError,
    ModelError,
    ResourceError,
    ServerError,
    format_exception,
    create_error_response,
    wrap_exception
)


class TestMCPXGBoostException:
    """Test the base MCPXGBoostException class."""

    def test_basic_exception_creation(self):
        """Test creating a basic exception."""
        exc = MCPXGBoostException("Test error")
        
        assert str(exc) == "[UNKNOWN_ERROR] Test error"
        assert exc.error_code == "UNKNOWN_ERROR"
        assert exc.details == {}

    def test_exception_with_code_and_details(self):
        """Test creating an exception with error code and details."""
        details = {"param": "value", "number": 42}
        exc = MCPXGBoostException(
            "Test error",
            error_code="TEST_ERROR",
            details=details
        )
        
        assert str(exc) == "[TEST_ERROR] Test error | Details: {'param': 'value', 'number': 42}"
        assert exc.error_code == "TEST_ERROR"
        assert exc.details == details

    def test_exception_details(self):
        """Test exception details method."""
        exc = MCPXGBoostException(
            "Test error",
            error_code="TEST_ERROR",
            details={"key": "value"}
        )
        
        details_dict = exc.to_dict()
        assert details_dict["error_code"] == "TEST_ERROR"
        assert details_dict["message"] == "Test error"
        assert details_dict["details"] == {"key": "value"}


class TestConfigurationError:
    """Test the ConfigurationError class."""

    def test_configuration_error_creation(self):
        """Test creating a configuration error."""
        exc = ConfigurationError("Invalid config parameter")
        
        assert str(exc) == "[CONFIG_ERROR] Invalid config parameter"
        assert exc.error_code == "CONFIG_ERROR"

    def test_configuration_error_with_details(self):
        """Test configuration error with details."""
        details = {"parameter": "max_depth", "value": -1}
        exc = ConfigurationError(
            "Invalid parameter value",
            details=details
        )
        
        assert exc.details == details
        assert exc.error_code == "CONFIG_ERROR"


class TestDataError:
    """Test the DataError class."""

    def test_data_error_creation(self):
        """Test creating a data error."""
        exc = DataError("Invalid data format")
        
        assert str(exc) == "[DATA_ERROR] Invalid data format"
        assert exc.error_code == "DATA_ERROR"

    def test_data_error_with_details(self):
        """Test data error with detailed context."""
        context = {
            "expected_columns": ["feature1", "feature2"],
            "actual_columns": ["feature1"],
            "missing_columns": ["feature2"]
        }
        exc = DataError("Missing required columns", details=context)
        
        assert exc.details == context


class TestModelError:
    """Test the ModelError class."""

    def test_model_error_creation(self):
        """Test creating a model error."""
        exc = ModelError("Model training failed")
        
        assert str(exc) == "[MODEL_ERROR] Model training failed"
        assert exc.error_code == "MODEL_ERROR"

    def test_model_error_with_model_context(self):
        """Test model error with model-specific context."""
        context = {
            "model_name": "test_model",
            "algorithm": "xgboost",
            "parameters": {"max_depth": 6}
        }
        exc = ModelError("Training failed", details=context)
        
        assert exc.details == context


class TestResourceError:
    """Test the ResourceError class."""

    def test_resource_error_creation(self):
        """Test creating a resource error."""
        exc = ResourceError("Out of memory")
        
        assert str(exc) == "[RESOURCE_ERROR] Out of memory"
        assert exc.error_code == "RESOURCE_ERROR"

    def test_resource_error_with_usage_context(self):
        """Test resource error with usage context."""
        context = {
            "memory_used": 1024,
            "memory_limit": 512,
            "cpu_usage": 95.5
        }
        exc = ResourceError("Resource limit exceeded", details=context)
        
        assert exc.details == context


class TestServerError:
    """Test the ServerError class."""

    def test_server_error_creation(self):
        """Test creating a server error."""
        exc = ServerError("Server connection failed")
        
        assert str(exc) == "[SERVER_ERROR] Server connection failed"
        assert exc.error_code == "SERVER_ERROR"

    def test_server_error_with_network_context(self):
        """Test server error with network context."""
        context = {
            "host": "localhost",
            "port": 8000,
            "timeout": 30
        }
        exc = ServerError("Connection timeout", details=context)
        
        assert exc.details == context


class TestUtilityFunctions:
    """Test utility functions for error handling."""

    def test_format_exception(self):
        """Test exception formatting."""
        exc = DataError(
            "Invalid data",
            details={"rows": 100, "columns": 5}
        )
        
        message = format_exception(exc)
        
        assert "Invalid data" in message
        assert "DATA_ERROR" in message

    def test_create_error_response(self):
        """Test error response creation."""
        exc = ModelError("Training failed")
        
        response = create_error_response(exc)
        
        assert response["success"] is False
        assert response["error_code"] == "MODEL_ERROR"
        assert response["message"] == "Training failed"
        # timestamp field might not be present in all implementations
        assert "details" in response

    def test_wrap_exception_with_mcp_exception(self):
        """Test wrapping an existing MCP exception."""
        original_exc = DataError("Original error")
        
        wrapped = wrap_exception(original_exc, "Additional context")
        
        # Should return a wrapped exception
        assert isinstance(wrapped, MCPXGBoostException)
        assert "Original error" in str(wrapped)

    def test_wrap_exception_with_standard_exception(self):
        """Test wrapping a standard Python exception."""
        original_exc = ValueError("Invalid value")
        
        wrapped = wrap_exception(original_exc, "Wrapped context")
        
        assert isinstance(wrapped, MCPXGBoostException)
        assert "Invalid value" in str(wrapped)
        assert "Wrapped context" in str(wrapped)

    def test_wrap_exception_with_context(self):
        """Test wrapping exception with additional context."""
        original_exc = KeyError("missing_key")
        context = {"operation": "data_access", "key": "missing_key"}
        
        wrapped = wrap_exception("data access", original_exc, context)
        
        assert isinstance(wrapped, MCPXGBoostException)
        assert wrapped.details == context
        assert "data access" in str(wrapped) 