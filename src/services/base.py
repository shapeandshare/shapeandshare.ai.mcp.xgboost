"""Abstract Base Classes for ML Service Architecture.

This module provides abstract base classes and interfaces for creating
extensible ML services in the MCP XGBoost server application. It defines
common patterns and contracts for ML operations.

The service abstraction supports:
- Standardized ML service interfaces
- Pluggable model implementations
- Consistent error handling and logging
- Resource management and cleanup
- Model lifecycle management
- Performance monitoring and metrics

Classes
-------
MLService
    Abstract base class for ML services
ModelManager
    Abstract base class for model management
DataProcessor
    Abstract base class for data processing
ValidationMixin
    Mixin for data validation capabilities
PersistenceMixin
    Mixin for model persistence capabilities
MetricsMixin
    Mixin for metrics collection capabilities
ServiceRegistry
    Registry for managing multiple ML services

Notes
-----
The service abstraction is designed to be:
- Extensible for different ML frameworks
- Consistent across all service implementations
- Type-safe with proper interface definitions
- Performance-aware with resource management
- Observable with comprehensive logging

Design Patterns
---------------
- Abstract Factory: For creating ML services
- Strategy: For different model implementations
- Observer: For metrics and monitoring
- Template Method: For common operation patterns
- Dependency Injection: For service configuration

Examples
--------
Implementing a custom ML service:

    >>> from src.services.base import MLService
    >>> class CustomMLService(MLService):
    ...     def train_model(self, name, data, params):
    ...         # Custom implementation
    ...         pass

Using the service registry:

    >>> from src.services.base import ServiceRegistry
    >>> registry = ServiceRegistry()
    >>> registry.register("xgboost", XGBoostService)
    >>> service = registry.get_service("xgboost")

See Also
--------
src.services.xgboost : XGBoost service implementation
src.config : Configuration management
src.exceptions : Custom exception handling
"""

import abc
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from fastmcp import FastMCP

from ..config import get_config
from ..logging_config import get_logger, log_request_context
from ..validation import DataValidationResult, validate_prediction_data, validate_training_data


class ServiceMetrics:
    """Metrics collection for ML services.

    Attributes
    ----------
    operation_counts : Dict[str, int]
        Count of operations by type
    operation_times : Dict[str, List[float]]
        Execution times for operations
    error_counts : Dict[str, int]
        Error counts by type
    resource_usage : Dict[str, Any]
        Resource usage statistics
    """

    def __init__(self):
        """Initialize service metrics."""
        self.operation_counts = {}
        self.operation_times = {}
        self.error_counts = {}
        self.resource_usage = {"memory_peak": 0, "models_in_memory": 0, "total_requests": 0}
        self._start_time = datetime.utcnow()

    def record_operation(self, operation: str, duration: float, success: bool = True):
        """Record an operation execution.

        Parameters
        ----------
        operation : str
            Name of the operation
        duration : float
            Execution time in seconds
        success : bool, optional
            Whether the operation succeeded (default: True)
        """
        # Count operations
        self.operation_counts[operation] = self.operation_counts.get(operation, 0) + 1

        # Record timing
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        self.operation_times[operation].append(duration)

        # Keep only last 1000 times to prevent memory growth
        if len(self.operation_times[operation]) > 1000:
            self.operation_times[operation] = self.operation_times[operation][-1000:]

        # Record errors
        if not success:
            self.error_counts[operation] = self.error_counts.get(operation, 0) + 1

        # Update resource usage
        self.resource_usage["total_requests"] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary.

        Returns
        -------
        Dict[str, Any]
            Metrics summary
        """
        uptime = (datetime.utcnow() - self._start_time).total_seconds()

        # Calculate average times
        avg_times = {}
        for operation, times in self.operation_times.items():
            if times:
                avg_times[operation] = {
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "count": len(times),
                }

        return {
            "uptime_seconds": uptime,
            "operation_counts": self.operation_counts,
            "average_times": avg_times,
            "error_counts": self.error_counts,
            "resource_usage": self.resource_usage,
            "error_rate": sum(self.error_counts.values()) / max(sum(self.operation_counts.values()), 1),
        }


class MLService(abc.ABC):
    """Abstract base class for ML services.

    This class defines the interface that all ML services must implement.
    It provides common functionality for model management, validation,
    and lifecycle operations.

    Attributes
    ----------
    name : str
        Service name identifier
    mcp : FastMCP
        MCP server instance
    config : AppConfig
        Application configuration
    logger : StructuredLogger
        Logger instance
    metrics : ServiceMetrics
        Metrics collector
    """

    def __init__(self, name: str, mcp: FastMCP):
        """Initialize the ML service.

        Parameters
        ----------
        name : str
            Service name identifier
        mcp : FastMCP
            MCP server instance
        """
        self.name = name
        self.mcp = mcp
        self.config = get_config()
        self.logger = get_logger(f"{__name__}.{name}")
        self.metrics = ServiceMetrics()
        self._initialized = False
        self._shutdown = False

    async def initialize(self):
        """Initialize the service.

        This method should be called before using the service.
        Subclasses can override this to perform custom initialization.
        """
        if self._initialized:
            return

        self.logger.info(f"Initializing ML service: {self.name}")
        await self._setup_service()
        self._register_tools()
        self._initialized = True

        self.logger.info(f"ML service initialized: {self.name}")

    async def shutdown(self):
        """Shutdown the service.

        This method should be called when the service is no longer needed.
        Subclasses can override this to perform custom cleanup.
        """
        if self._shutdown:
            return

        self.logger.info(f"Shutting down ML service: {self.name}")
        await self._cleanup_service()
        self._shutdown = True

        self.logger.info(f"ML service shutdown complete: {self.name}")

    @abc.abstractmethod
    async def _setup_service(self):
        """Setup service-specific resources.

        This method should be implemented by subclasses to perform
        any service-specific initialization.
        """
        pass

    @abc.abstractmethod
    async def _cleanup_service(self):
        """Cleanup service-specific resources.

        This method should be implemented by subclasses to perform
        any service-specific cleanup.
        """
        pass

    @abc.abstractmethod
    def _register_tools(self):
        """Register MCP tools for this service.

        This method should be implemented by subclasses to register
        their specific MCP tools.
        """
        pass

    @abc.abstractmethod
    async def train_model(self, model_name: str, data: str, target_column: str, **parameters) -> str:
        """Train a model with the given data and parameters.

        Parameters
        ----------
        model_name : str
            Name to save the model under
        data : str
            JSON string containing training data
        target_column : str
            Name of the target column
        **parameters
            Additional model parameters

        Returns
        -------
        str
            Training result message
        """
        pass

    @abc.abstractmethod
    async def predict(self, model_name: str, data: str) -> str:
        """Make predictions using a trained model.

        Parameters
        ----------
        model_name : str
            Name of the model to use
        data : str
            JSON string containing prediction data

        Returns
        -------
        str
            Prediction results
        """
        pass

    @abc.abstractmethod
    async def get_model_info(self, model_name: str) -> str:
        """Get information about a trained model.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        str
            Model information
        """
        pass

    @abc.abstractmethod
    async def list_models(self) -> str:
        """List all available models.

        Returns
        -------
        str
            List of available models
        """
        pass

    @asynccontextmanager
    async def _operation_context(self, operation: str, **context):
        """Context manager for operation tracking.

        Parameters
        ----------
        operation : str
            Name of the operation
        **context
            Additional context information
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        success = True

        try:
            with log_request_context(request_id=request_id, operation=operation, service=self.name, **context):
                self.logger.info(f"Starting operation: {operation}", extra=context)
                yield request_id

        except Exception as e:
            success = False
            self.logger.error(
                f"Operation failed: {operation}",
                exc_info=True,
                extra={"error": str(e), "request_id": request_id, **context},
            )
            raise

        finally:
            duration = time.time() - start_time
            self.metrics.record_operation(operation, duration, success)

            if success:
                self.logger.info(
                    f"Operation completed: {operation}",
                    extra={"duration": duration, "request_id": request_id, **context},
                )

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics.

        Returns
        -------
        Dict[str, Any]
            Service metrics
        """
        return {
            "service_name": self.name,
            "initialized": self._initialized,
            "shutdown": self._shutdown,
            **self.metrics.get_summary(),
        }


class ModelManager(abc.ABC):
    """Abstract base class for model management.

    This class defines the interface for managing ML models,
    including storage, retrieval, and lifecycle operations.
    """

    @abc.abstractmethod
    async def save_model(self, name: str, model: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save a model with metadata.

        Parameters
        ----------
        name : str
            Model name
        model : Any
            Model object to save
        metadata : Dict[str, Any], optional
            Model metadata

        Returns
        -------
        bool
            True if successful
        """
        pass

    @abc.abstractmethod
    async def load_model(self, name: str) -> Any:
        """Load a model by name.

        Parameters
        ----------
        name : str
            Model name

        Returns
        -------
        Any
            Loaded model object
        """
        pass

    @abc.abstractmethod
    async def delete_model(self, name: str) -> bool:
        """Delete a model by name.

        Parameters
        ----------
        name : str
            Model name

        Returns
        -------
        bool
            True if successful
        """
        pass

    @abc.abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models.

        Returns
        -------
        List[Dict[str, Any]]
            List of model information
        """
        pass


class DataProcessor(abc.ABC):
    """Abstract base class for data processing.

    This class defines the interface for data processing operations
    such as validation, transformation, and feature engineering.
    """

    @abc.abstractmethod
    async def validate_data(self, data: Dict[str, Any], schema: str) -> DataValidationResult:
        """Validate data against a schema.

        Parameters
        ----------
        data : Dict[str, Any]
            Data to validate
        schema : str
            Schema identifier

        Returns
        -------
        DataValidationResult
            Validation result
        """
        pass

    @abc.abstractmethod
    async def transform_data(self, data: Dict[str, Any], transformations: List[str]) -> Dict[str, Any]:
        """Apply transformations to data.

        Parameters
        ----------
        data : Dict[str, Any]
            Data to transform
        transformations : List[str]
            List of transformation names

        Returns
        -------
        Dict[str, Any]
            Transformed data
        """
        pass


class ValidationMixin:
    """Mixin for data validation capabilities.

    This mixin provides common data validation functionality
    that can be used by ML services.
    """

    def validate_training_data(self, data: Dict[str, Any], target_column: str) -> DataValidationResult:
        """Validate training data."""
        return validate_training_data(data, target_column)

    def validate_prediction_data(
        self, data: Dict[str, Any], expected_features: Optional[List[str]] = None
    ) -> DataValidationResult:
        """Validate prediction data."""
        return validate_prediction_data(data, expected_features)


class PersistenceMixin:
    """Mixin for model persistence capabilities.

    This mixin provides common model persistence functionality
    that can be used by ML services.
    """

    def __init__(self, *args, **kwargs):
        """Initialize persistence mixin."""
        super().__init__(*args, **kwargs)
        self._models = {}  # In-memory model storage

    def _store_model(self, name: str, model: Any, metadata: Optional[Dict[str, Any]] = None):
        """Store a model in memory."""
        self._models[name] = {
            "model": model,
            "metadata": metadata or {},
            "created_at": datetime.utcnow(),
            "access_count": 0,
        }

    def _retrieve_model(self, name: str) -> Optional[Any]:
        """Retrieve a model from memory."""
        if name in self._models:
            self._models[name]["access_count"] += 1
            return self._models[name]["model"]
        return None

    def _delete_model(self, name: str) -> bool:
        """Delete a model from memory."""
        if name in self._models:
            del self._models[name]
            return True
        return False

    def _list_models(self) -> List[Dict[str, Any]]:
        """List all models in memory."""
        return [
            {
                "name": name,
                "metadata": info["metadata"],
                "created_at": info["created_at"],
                "access_count": info["access_count"],
            }
            for name, info in self._models.items()
        ]


class MetricsMixin:
    """Mixin for metrics collection capabilities.

    This mixin provides common metrics collection functionality
    that can be used by ML services.
    """

    def __init__(self, *args, **kwargs):
        """Initialize metrics mixin."""
        super().__init__(*args, **kwargs)
        self.metrics = ServiceMetrics()

    def record_operation(self, operation: str, duration: float, success: bool = True):
        """Record an operation for metrics."""
        self.metrics.record_operation(operation, duration, success)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return self.metrics.get_summary()


class ServiceRegistry:
    """Registry for managing multiple ML services.

    This class provides a centralized registry for ML services,
    allowing for dynamic service discovery and management.

    Attributes
    ----------
    services : Dict[str, Type[MLService]]
        Registered service classes
    instances : Dict[str, MLService]
        Active service instances
    """

    def __init__(self):
        """Initialize the service registry."""
        self.services: Dict[str, Type[MLService]] = {}
        self.instances: Dict[str, MLService] = {}
        self.logger = get_logger(__name__)

    def register(self, name: str, service_class: Type[MLService]):
        """Register a service class.

        Parameters
        ----------
        name : str
            Service name
        service_class : Type[MLService]
            Service class to register
        """
        if not issubclass(service_class, MLService):
            raise ValueError("Service class must inherit from MLService")

        self.services[name] = service_class
        self.logger.info("Registered service", extra={"name": name})

    async def unregister(self, name: str):
        """Unregister a service.

        Parameters
        ----------
        name : str
            Service name to unregister
        """
        if name in self.services:
            del self.services[name]

        if name in self.instances:
            # Shutdown the instance
            await self.instances[name].shutdown()
            del self.instances[name]

        self.logger.info(f"Unregistered service: {name}")

    async def get_service(self, name: str, mcp: FastMCP) -> MLService:
        """Get a service instance.

        Parameters
        ----------
        name : str
            Service name
        mcp : FastMCP
            MCP server instance

        Returns
        -------
        MLService
            Service instance
        """
        if name not in self.services:
            raise ValueError(f"Service not registered: {name}")

        if name not in self.instances:
            # Create and initialize new instance
            service_class = self.services[name]
            instance = service_class(name, mcp)
            await instance.initialize()
            self.instances[name] = instance

            self.logger.info(f"Created service instance: {name}")

        return self.instances[name]

    def list_services(self) -> List[str]:
        """List all registered services.

        Returns
        -------
        List[str]
            List of service names
        """
        return list(self.services.keys())

    def list_instances(self) -> List[str]:
        """List all active service instances.

        Returns
        -------
        List[str]
            List of active service names
        """
        return list(self.instances.keys())

    async def shutdown_all(self):
        """Shutdown all active service instances."""
        for name, instance in self.instances.items():
            await instance.shutdown()
            self.logger.info(f"Shutdown service instance: {name}")

        self.instances.clear()


# Global service registry
service_registry = ServiceRegistry()
