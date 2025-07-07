"""Custom Exception Classes for MCP XGBoost Server.

This module defines custom exception classes for better error handling,
debugging, and user feedback in the MCP XGBoost server application.

The exceptions are organized into logical categories:
- Configuration and setup errors
- Data validation and processing errors
- Model training and prediction errors
- Resource and storage errors
- Server and network errors

Exception Hierarchy
------------------
MCPXGBoostException (base)
├── ConfigurationError
│   ├── InvalidConfigurationError
│   └── MissingConfigurationError
├── DataError
│   ├── DataValidationError
│   ├── DataFormatError
│   └── DataSizeError
├── ModelError
│   ├── ModelNotFoundError
│   ├── ModelTrainingError
│   ├── ModelPredictionError
│   └── ModelPersistenceError
├── ResourceError
│   ├── MemoryLimitError
│   ├── ResourceExhaustedError
│   └── StorageError
└── ServerError
    ├── RequestTimeoutError
    ├── ConcurrencyLimitError
    └── NetworkError

Classes
-------
MCPXGBoostException
    Base exception class for all application-specific errors
ConfigurationError
    Base class for configuration-related errors
DataError
    Base class for data-related errors
ModelError
    Base class for model-related errors
ResourceError
    Base class for resource-related errors
ServerError
    Base class for server-related errors

Functions
---------
format_exception
    Format exception for user-friendly display
create_error_response
    Create standardized error response

Notes
-----
All custom exceptions include detailed error messages, error codes,
and contextual information to help with debugging and user feedback.

The exceptions follow a consistent structure with:
- Error code for programmatic handling
- User-friendly message for display
- Technical details for debugging
- Contextual information where relevant

Examples
--------
Raise a custom exception:

    >>> from src.exceptions import DataValidationError
    >>> raise DataValidationError("Invalid data format", details={"column": "target"})

Handle exceptions with context:

    >>> try:
    ...     # Some operation
    ...     pass
    ... except MCPXGBoostException as e:
    ...     print(f"Error {e.error_code}: {e.message}")
    ...     print(f"Details: {e.details}")

See Also
--------
src.config : Configuration management
src.utils : Utility functions
"""

from typing import Any, Dict, Optional


class MCPXGBoostException(Exception):
    """Base exception class for all MCP XGBoost application errors.

    This is the base class for all custom exceptions in the application.
    It provides a consistent interface for error handling and reporting.

    Attributes
    ----------
    error_code : str
        Unique error code for programmatic handling
    message : str
        User-friendly error message
    details : Dict[str, Any]
        Additional error details and context
    cause : Optional[Exception]
        Original exception that caused this error
    """

    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """Initialize the exception.

        Parameters
        ----------
        message : str
            User-friendly error message
        error_code : str, optional
            Unique error code (default: "UNKNOWN_ERROR")
        details : Dict[str, Any], optional
            Additional error details and context
        cause : Exception, optional
            Original exception that caused this error
        """
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        """Return string representation of the exception."""
        result = f"[{self.error_code}] {self.message}"
        if self.details:
            result += f" | Details: {self.details}"
        if self.cause:
            result += f" | Caused by: {self.cause}"
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
            "type": self.__class__.__name__,
        }


# Configuration Errors
class ConfigurationError(MCPXGBoostException):
    """Base class for configuration-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration values are invalid."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="INVALID_CONFIG", **kwargs)


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="MISSING_CONFIG", **kwargs)


# Data Errors
class DataError(MCPXGBoostException):
    """Base class for data-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="DATA_ERROR", **kwargs)


class DataValidationError(DataError):
    """Raised when data validation fails."""

    def __init__(self, message: str, **kwargs):
        kwargs.pop("error_code", None)  # Remove error_code from kwargs to avoid duplicate
        super().__init__(message, **kwargs)
        self.error_code = "DATA_VALIDATION_ERROR"


class DataFormatError(DataError):
    """Raised when data format is invalid or unsupported."""

    def __init__(self, message: str, **kwargs):
        kwargs.pop("error_code", None)  # Remove error_code from kwargs to avoid duplicate
        super().__init__(message, **kwargs)
        self.error_code = "DATA_FORMAT_ERROR"


class DataSizeError(DataError):
    """Raised when data size exceeds limits."""

    def __init__(self, message: str, **kwargs):
        kwargs.pop("error_code", None)  # Remove error_code from kwargs to avoid duplicate
        super().__init__(message, **kwargs)
        self.error_code = "DATA_SIZE_ERROR"


# Model Errors
class ModelError(MCPXGBoostException):
    """Base class for model-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="MODEL_ERROR", **kwargs)


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found."""

    def __init__(self, message: str, **kwargs):
        kwargs.pop("error_code", None)
        super().__init__(message, **kwargs)
        self.error_code = "MODEL_NOT_FOUND"


class ModelTrainingError(ModelError):
    """Raised when model training fails."""

    def __init__(self, message: str, **kwargs):
        kwargs.pop("error_code", None)
        super().__init__(message, **kwargs)
        self.error_code = "MODEL_TRAINING_ERROR"


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""

    def __init__(self, message: str, **kwargs):
        kwargs.pop("error_code", None)
        super().__init__(message, **kwargs)
        self.error_code = "MODEL_PREDICTION_ERROR"


class ModelPersistenceError(ModelError):
    """Raised when model persistence operations fail."""

    def __init__(self, message: str, **kwargs):
        kwargs.pop("error_code", None)
        super().__init__(message, **kwargs)
        self.error_code = "MODEL_PERSISTENCE_ERROR"


# Resource Errors
class ResourceError(MCPXGBoostException):
    """Base class for resource-related errors."""

    def __init__(self, message: str, error_code: str = "RESOURCE_ERROR", **kwargs):
        super().__init__(message, error_code=error_code, **kwargs)


class MemoryLimitError(ResourceError):
    """Raised when memory usage exceeds limits."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="MEMORY_LIMIT_ERROR", **kwargs)


class ResourceExhaustedError(ResourceError):
    """Raised when system resources are exhausted."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="RESOURCE_EXHAUSTED", **kwargs)


class StorageError(ResourceError):
    """Raised when storage operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="STORAGE_ERROR", **kwargs)


# Server Errors
class ServerError(MCPXGBoostException):
    """Base class for server-related errors."""

    def __init__(self, message: str, error_code: str = "SERVER_ERROR", **kwargs):
        super().__init__(message, error_code=error_code, **kwargs)


class RequestTimeoutError(ServerError):
    """Raised when request processing times out."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="REQUEST_TIMEOUT", **kwargs)


class ConcurrencyLimitError(ServerError):
    """Raised when concurrency limits are exceeded."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="CONCURRENCY_LIMIT", **kwargs)


class NetworkError(ServerError):
    """Raised when network operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="NETWORK_ERROR", **kwargs)


# Utility Functions
def format_exception(exception: Exception) -> str:
    """Format exception for user-friendly display.

    Parameters
    ----------
    exception : Exception
        The exception to format

    Returns
    -------
    str
        Formatted exception message

    Examples
    --------
    Format a custom exception:

        >>> from src.exceptions import DataValidationError
        >>> exc = DataValidationError("Invalid data", details={"column": "target"})
        >>> formatted = format_exception(exc)
        >>> print(formatted)
        [DATA_VALIDATION_ERROR] Invalid data | Details: {'column': 'target'}

    Format a standard exception:

        >>> exc = ValueError("Invalid value")
        >>> formatted = format_exception(exc)
        >>> print(formatted)
        ValueError: Invalid value
    """
    if isinstance(exception, MCPXGBoostException):
        return str(exception)

    return f"{exception.__class__.__name__}: {str(exception)}"


def create_error_response(exception: Exception, include_technical_details: bool = False) -> Dict[str, Any]:
    """Create standardized error response dictionary.

    Parameters
    ----------
    exception : Exception
        The exception to convert to response
    include_technical_details : bool, optional
        Whether to include technical details (default: False)

    Returns
    -------
    Dict[str, Any]
        Standardized error response dictionary

    Examples
    --------
    Create error response for API:

        >>> from src.exceptions import ModelNotFoundError
        >>> exc = ModelNotFoundError("Model 'test' not found")
        >>> response = create_error_response(exc)
        >>> print(response)
        {
            'success': False,
            'error_code': 'MODEL_NOT_FOUND',
            'message': "Model 'test' not found",
            'details': {}
        }
    """
    if isinstance(exception, MCPXGBoostException):
        response = {
            "success": False,
            "error_code": exception.error_code,
            "message": exception.message,
            "details": exception.details,
        }

        if include_technical_details:
            response["technical_details"] = {
                "exception_type": exception.__class__.__name__,
                "cause": str(exception.cause) if exception.cause else None,
                "full_message": str(exception),
            }
    else:
        response = {
            "success": False,
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "details": {},
        }

        if include_technical_details:
            response["technical_details"] = {
                "exception_type": exception.__class__.__name__,
                "full_message": str(exception),
            }

    return response


def wrap_exception(
    operation: str, original_exception: Exception, context: Optional[Dict[str, Any]] = None
) -> MCPXGBoostException:
    """Wrap a standard exception in a custom exception with context.

    Parameters
    ----------
    operation : str
        Description of the operation that failed
    original_exception : Exception
        The original exception that occurred
    context : Dict[str, Any], optional
        Additional context information

    Returns
    -------
    MCPXGBoostException
        Wrapped exception with context

    Examples
    --------
    Wrap a JSON parsing error:

        >>> import json
        >>> try:
        ...     json.loads("invalid json")
        ... except json.JSONDecodeError as e:
        ...     wrapped = wrap_exception("parsing data", e, {"data_type": "json"})
        ...     raise wrapped
    """
    # Determine appropriate exception type based on original exception
    if isinstance(original_exception, (ValueError, TypeError)):
        if "json" in str(original_exception).lower():
            return DataFormatError(
                f"Failed to {operation}: {str(original_exception)}", details=context or {}, cause=original_exception
            )

        return DataValidationError(
            f"Failed to {operation}: {str(original_exception)}", details=context or {}, cause=original_exception
        )

    if isinstance(original_exception, (FileNotFoundError, PermissionError)):
        return StorageError(
            f"Failed to {operation}: {str(original_exception)}", details=context or {}, cause=original_exception
        )

    if isinstance(original_exception, (MemoryError, OSError)):
        return ResourceError(
            f"Failed to {operation}: {str(original_exception)}", details=context or {}, cause=original_exception
        )

    return MCPXGBoostException(
        f"Failed to {operation}: {str(original_exception)}",
        error_code="WRAPPED_ERROR",
        details=context or {},
        cause=original_exception,
    )
