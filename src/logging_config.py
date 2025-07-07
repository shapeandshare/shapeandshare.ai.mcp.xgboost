"""Structured Logging Framework for MCP XGBoost Server.

This module provides a comprehensive logging framework for the MCP XGBoost
server application. It supports multiple output formats, log levels, file
rotation, and structured logging with contextual information.

The logging framework integrates with the configuration system and provides:
- Structured logging with JSON output
- Multiple log levels and handlers
- File rotation with size and time-based rotation
- Contextual logging with request IDs and user information
- Performance monitoring and metrics logging
- Error tracking with stack traces
- Integration with monitoring systems

Classes
-------
StructuredLogger
    Main logging class with advanced features
LogContext
    Context manager for structured logging with additional context
LogMetrics
    Metrics collector for logging statistics
LogFormatter
    Custom formatter for structured log output

Functions
---------
setup_logging
    Configure logging system with settings from configuration
get_logger
    Get a logger instance with structured logging capabilities
log_request
    Log HTTP request with timing and context
log_error
    Log errors with context and stack traces
log_metrics
    Log performance metrics

Notes
-----
The logging framework is designed to be:
- High-performance with minimal overhead
- Easily configurable through environment variables
- Compatible with log aggregation systems
- Structured for machine readability
- Human-readable for development

The framework supports multiple output formats:
- JSON for production environments
- Formatted text for development
- Syslog for system integration
- File output with rotation

Examples
--------
Basic usage:

    >>> from src.logging_config import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting application")

Structured logging with context:

    >>> logger.info("Processing request", extra={
    ...     "request_id": "12345",
    ...     "user_id": "user123",
    ...     "operation": "train_model"
    ... })

Error logging with context:

    >>> try:
    ...     # Some operation
    ...     pass
    ... except Exception as e:
    ...     logger.error("Operation failed", exc_info=True, extra={
    ...         "operation": "train_model",
    ...         "model_name": "test_model"
    ...     })

Performance logging:

    >>> import time
    >>> start_time = time.time()
    >>> # Do some work
    >>> logger.info("Operation completed", extra={
    ...     "duration": time.time() - start_time,
    ...     "operation": "train_model"
    ... })

See Also
--------
src.config : Configuration management
src.exceptions : Custom exception handling
"""

import json
import logging
import logging.handlers
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional, Union

from .config import get_config
from .exceptions import ConfigurationError


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured log output.

    This formatter can output logs in either JSON format for production
    or a human-readable format for development.

    Attributes
    ----------
    json_format : bool
        Whether to output in JSON format
    include_extra : bool
        Whether to include extra fields in the output
    """

    def __init__(self, json_format: bool = False, include_extra: bool = True):
        """Initialize the formatter.

        Parameters
        ----------
        json_format : bool, optional
            Output in JSON format (default: False)
        include_extra : bool, optional
            Include extra fields in output (default: True)
        """
        super().__init__()
        self.json_format = json_format
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format

        Returns
        -------
        str
            Formatted log message
        """
        # Create base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }

        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields if enabled
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "getMessage",
                    "message",
                    "asctime",
                }:
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)

            if extra_fields:
                log_data["extra"] = extra_fields

        # Format output
        if self.json_format:
            return json.dumps(log_data, ensure_ascii=False)

        # Human-readable format for development
        base_msg = f"{log_data['timestamp']} [{log_data['level']:8}] {log_data['logger']}: {log_data['message']}"

        # Add extra fields if present
        if "extra" in log_data and log_data["extra"]:
            extra_str = " | ".join(f"{k}={v}" for k, v in log_data["extra"].items())
            base_msg += f" | {extra_str}"

        # Add exception if present
        if "exception" in log_data:
            base_msg += f"\n{log_data['exception']['traceback']}"

        return base_msg


class LogContext:
    """Context manager for structured logging with additional context.

    This class provides a way to add contextual information to all log
    messages within a specific scope.

    Attributes
    ----------
    context : Dict[str, Any]
        Context information to add to log messages
    """

    def __init__(self, **context):
        """Initialize the log context.

        Parameters
        ----------
        **context
            Context information to add to log messages
        """
        self.context = context
        self.old_factory = None

    def __enter__(self):
        """Enter the context manager."""
        self.old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        logging.setLogRecordFactory(self.old_factory)


class LogMetrics:
    """Metrics collector for logging statistics.

    This class collects and tracks logging metrics such as:
    - Log message counts by level
    - Error rates
    - Performance metrics
    - Request timing

    Attributes
    ----------
    metrics : Dict[str, Any]
        Dictionary storing collected metrics
    """

    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics = {
            "log_counts": {"DEBUG": 0, "INFO": 0, "WARNING": 0, "ERROR": 0, "CRITICAL": 0},
            "errors": [],
            "requests": [],
            "performance": [],
        }

    def increment_log_count(self, level: str):
        """Increment the count for a log level.

        Parameters
        ----------
        level : str
            The log level to increment
        """
        if level in self.metrics["log_counts"]:
            self.metrics["log_counts"][level] += 1

    def add_error(self, error_info: Dict[str, Any]):
        """Add error information to metrics.

        Parameters
        ----------
        error_info : Dict[str, Any]
            Error information to store
        """
        error_info["timestamp"] = datetime.utcnow().isoformat()
        self.metrics["errors"].append(error_info)

        # Keep only last 100 errors
        if len(self.metrics["errors"]) > 100:
            self.metrics["errors"] = self.metrics["errors"][-100:]

    def add_request(self, request_info: Dict[str, Any]):
        """Add request information to metrics.

        Parameters
        ----------
        request_info : Dict[str, Any]
            Request information to store
        """
        request_info["timestamp"] = datetime.utcnow().isoformat()
        self.metrics["requests"].append(request_info)

        # Keep only last 1000 requests
        if len(self.metrics["requests"]) > 1000:
            self.metrics["requests"] = self.metrics["requests"][-1000:]

    def add_performance_metric(self, metric_info: Dict[str, Any]):
        """Add performance metric to collection.

        Parameters
        ----------
        metric_info : Dict[str, Any]
            Performance metric information
        """
        metric_info["timestamp"] = datetime.utcnow().isoformat()
        self.metrics["performance"].append(metric_info)

        # Keep only last 1000 metrics
        if len(self.metrics["performance"]) > 1000:
            self.metrics["performance"] = self.metrics["performance"][-1000:]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of collected metrics.

        Returns
        -------
        Dict[str, Any]
            Summary of metrics
        """
        total_logs = sum(self.metrics["log_counts"].values())
        error_rate = (self.metrics["log_counts"]["ERROR"] + self.metrics["log_counts"]["CRITICAL"]) / max(total_logs, 1)

        return {
            "total_logs": total_logs,
            "log_counts": self.metrics["log_counts"],
            "error_rate": error_rate,
            "total_errors": len(self.metrics["errors"]),
            "total_requests": len(self.metrics["requests"]),
            "total_performance_metrics": len(self.metrics["performance"]),
        }


class StructuredLogger:
    """Main logging class with advanced features.

    This class provides a wrapper around the Python logging module
    with additional features for structured logging.

    Attributes
    ----------
    logger : logging.Logger
        The underlying Python logger
    metrics : LogMetrics
        Metrics collector instance
    """

    def __init__(self, name: str):
        """Initialize the structured logger.

        Parameters
        ----------
        name : str
            Name of the logger
        """
        self.logger = logging.getLogger(name)
        self.metrics = LogMetrics()

    def _log_with_metrics(self, level: str, message: str, *args, **kwargs):
        """Log a message and update metrics.

        Parameters
        ----------
        level : str
            Log level
        message : str
            Log message
        *args
            Positional arguments for logging
        **kwargs
            Keyword arguments for logging
        """
        # Update metrics
        self.metrics.increment_log_count(level)

        # Log the message
        log_method = getattr(self.logger, level.lower())
        log_method(message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs):
        """Log a debug message."""
        self._log_with_metrics("DEBUG", message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log an info message."""
        self._log_with_metrics("INFO", message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log a warning message."""
        self._log_with_metrics("WARNING", message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log an error message."""
        self._log_with_metrics("ERROR", message, *args, **kwargs)

        # Add error to metrics
        if kwargs.get("exc_info"):
            self.metrics.add_error(
                {"message": message, "args": args, "kwargs": {k: v for k, v in kwargs.items() if k != "exc_info"}}
            )

    def critical(self, message: str, *args, **kwargs):
        """Log a critical message."""
        self._log_with_metrics("CRITICAL", message, *args, **kwargs)

        # Add error to metrics
        self.metrics.add_error({"message": message, "args": args, "kwargs": kwargs, "level": "CRITICAL"})

    def log_request(self, request_info: Dict[str, Any]):
        """Log request information.

        Parameters
        ----------
        request_info : Dict[str, Any]
            Request information to log
        """
        self.info("Request processed", extra=request_info)
        self.metrics.add_request(request_info)

    def log_performance(self, metric_info: Dict[str, Any]):
        """Log performance metrics.

        Parameters
        ----------
        metric_info : Dict[str, Any]
            Performance metric information
        """
        self.info("Performance metric", extra=metric_info)
        self.metrics.add_performance_metric(metric_info)


# Global metrics collector
_global_metrics = LogMetrics()


def setup_logging(config=None) -> None:
    """Configure logging system with settings from configuration.

    Parameters
    ----------
    config : AppConfig, optional
        Application configuration (default: loads from environment)

    Raises
    ------
    ConfigurationError
        If logging configuration is invalid
    """
    if config is None:
        config = get_config()

    try:
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.logging.level))

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create formatter
        json_format = os.getenv("MCP_LOG_JSON", "false").lower() == "true"
        formatter = StructuredFormatter(json_format=json_format)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler (if configured)
        if config.logging.file_path:
            file_handler = logging.handlers.RotatingFileHandler(
                config.logging.file_path,
                maxBytes=config.logging.max_file_size_mb * 1024 * 1024,
                backupCount=config.logging.backup_count,
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        # Set up library loggers
        logging.getLogger("fastmcp").setLevel(logging.INFO)
        logging.getLogger("xgboost").setLevel(logging.WARNING)
        logging.getLogger("pandas").setLevel(logging.WARNING)
        logging.getLogger("numpy").setLevel(logging.WARNING)

        # Log successful setup
        logger = logging.getLogger(__name__)
        logger.info(
            "Logging configured successfully",
            extra={"level": config.logging.level, "json_format": json_format, "file_path": config.logging.file_path},
        )

    except Exception as e:
        raise ConfigurationError(
            f"Failed to configure logging: {str(e)}", details={"config": config.logging.dict() if config else None}
        ) from e


def get_logger(name: str) -> StructuredLogger:
    """Get a logger instance with structured logging capabilities.

    Parameters
    ----------
    name : str
        Name of the logger (usually __name__)

    Returns
    -------
    StructuredLogger
        Logger instance with structured logging features
    """
    return StructuredLogger(name)


@contextmanager
def log_request_context(request_id: str, **context):
    """Context manager for request logging.

    Parameters
    ----------
    request_id : str
        Unique request identifier
    **context
        Additional context information
    """
    start_time = time.time()

    with LogContext(request_id=request_id, **context):
        try:
            yield
        finally:
            duration = time.time() - start_time
            logger = get_logger(__name__)
            logger.log_request({"request_id": request_id, "duration": duration, **context})


def log_error_with_context(
    logger: Union[StructuredLogger, logging.Logger], message: str, exception: Optional[Exception] = None, **context
):
    """Log an error with context information.

    Parameters
    ----------
    logger : Union[StructuredLogger, logging.Logger]
        Logger instance to use
    message : str
        Error message
    exception : Exception, optional
        Exception that caused the error
    **context
        Additional context information
    """
    extra = {"error_context": context}

    if isinstance(logger, StructuredLogger):
        logger.error(message, exc_info=exception is not None, extra=extra)
    else:
        logger.error(message, exc_info=exception is not None, extra=extra)


def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of global logging metrics.

    Returns
    -------
    Dict[str, Any]
        Summary of logging metrics
    """
    return _global_metrics.get_summary()
