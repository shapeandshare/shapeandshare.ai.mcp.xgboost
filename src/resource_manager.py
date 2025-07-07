"""Resource Management for MCP XGBoost Server.

This module provides resource management capabilities including memory monitoring,
request throttling, and resource limit enforcement for the MCP XGBoost server.

Classes
-------
ResourceManager
    Main resource management coordinator
MemoryManager
    Memory usage monitoring and limits
RequestThrottler
    Request rate limiting and concurrency control
ResourceMonitor
    System resource monitoring

Functions
---------
get_resource_manager
    Get global resource manager instance
check_memory_usage
    Check current memory usage
enforce_limits
    Enforce resource limits decorator

Notes
-----
This is a simplified implementation focusing on the most critical
resource management features for the application.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import psutil

from .config import get_config
from .exceptions import ConcurrencyLimitError, MemoryLimitError, ResourceError
from .logging_config import get_logger


class MemoryManager:
    """Memory usage monitoring and limits."""

    def __init__(self, max_memory_mb: int):
        """Initialize memory manager.

        Parameters
        ----------
        max_memory_mb : int
            Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        self.logger = get_logger(__name__)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage.

        Returns
        -------
        Dict[str, Any]
            Memory usage information
        """
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": (self.max_memory_mb - memory_info.rss / 1024 / 1024),
            "limit_mb": self.max_memory_mb,
        }

    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits.

        Returns
        -------
        bool
            True if within limits

        Raises
        ------
        MemoryLimitError
            If memory limit exceeded
        """
        usage = self.get_memory_usage()

        if usage["rss_mb"] > self.max_memory_mb:
            raise MemoryLimitError(
                f"Memory limit exceeded: {usage['rss_mb']:.1f}MB > {self.max_memory_mb}MB", details=usage
            )

        return True


class RequestThrottler:
    """Request rate limiting and concurrency control."""

    def __init__(self, max_concurrent: int, rate_limit: int = 100):
        """Initialize request throttler.

        Parameters
        ----------
        max_concurrent : int
            Maximum concurrent requests
        rate_limit : int, optional
            Maximum requests per minute (default: 100)
        """
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit
        self.current_requests = 0
        self.request_times: list[float] = []
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = get_logger(__name__)

    @asynccontextmanager
    async def acquire_request_slot(self):
        """Acquire a request slot with throttling.

        Raises
        ------
        ConcurrencyLimitError
            If too many concurrent requests
        """
        # Check rate limit
        now = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]

        if len(self.request_times) >= self.rate_limit:
            raise ConcurrencyLimitError(
                f"Rate limit exceeded: {len(self.request_times)} requests in last minute",
                details={"rate_limit": self.rate_limit, "current_rate": len(self.request_times)},
            )

        # Acquire semaphore for concurrency control
        async with self.semaphore:
            self.current_requests += 1
            self.request_times.append(now)

            try:
                yield
            finally:
                self.current_requests -= 1

    def get_status(self) -> Dict[str, Any]:
        """Get throttler status.

        Returns
        -------
        Dict[str, Any]
            Throttler status
        """
        now = time.time()
        recent_requests = [t for t in self.request_times if now - t < 60]

        return {
            "current_requests": self.current_requests,
            "max_concurrent": self.max_concurrent,
            "requests_last_minute": len(recent_requests),
            "rate_limit": self.rate_limit,
            "available_slots": self.max_concurrent - self.current_requests,
        }


class ResourceMonitor:
    """System resource monitoring."""

    def __init__(self):
        """Initialize resource monitor."""
        self.logger = get_logger(__name__)
        self.start_time = time.time()

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics.

        Returns
        -------
        Dict[str, Any]
            System statistics
        """
        try:
            # CPU information
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory information
            memory = psutil.virtual_memory()

            # Disk information
            disk = psutil.disk_usage("/")

            # Process information
            process = psutil.Process()
            process_memory = process.memory_info()

            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "load_avg": psutil.getloadavg() if hasattr(psutil, "getloadavg") else None,
                },
                "memory": {
                    "total_mb": memory.total / 1024 / 1024,
                    "available_mb": memory.available / 1024 / 1024,
                    "used_mb": memory.used / 1024 / 1024,
                    "percent": memory.percent,
                },
                "disk": {
                    "total_gb": disk.total / 1024 / 1024 / 1024,
                    "free_gb": disk.free / 1024 / 1024 / 1024,
                    "used_gb": disk.used / 1024 / 1024 / 1024,
                    "percent": (disk.used / disk.total) * 100,
                },
                "process": {
                    "memory_mb": process_memory.rss / 1024 / 1024,
                    "memory_percent": process.memory_percent(),
                    "cpu_percent": process.cpu_percent(),
                    "num_threads": process.num_threads(),
                },
                "uptime_seconds": time.time() - self.start_time,
            }

        except (psutil.Error, OSError, AttributeError) as e:
            self.logger.error(f"Failed to get system stats: {e}")
            return {"error": str(e)}


class ResourceManager:
    """Main resource management coordinator."""

    def __init__(self, config=None):
        """Initialize resource manager.

        Parameters
        ----------
        config : AppConfig, optional
            Application configuration
        """
        if config is None:
            config = get_config()

        self.config = config
        self.logger = get_logger(__name__)

        # Initialize components
        self.memory_manager = MemoryManager(config.resources.max_memory_mb)
        self.request_throttler = RequestThrottler(max_concurrent=config.server.max_concurrent_requests)
        self.monitor = ResourceMonitor()

        self.logger.info(
            "Resource manager initialized",
            extra={
                "max_memory_mb": config.resources.max_memory_mb,
                "max_concurrent": config.server.max_concurrent_requests,
            },
        )

    @asynccontextmanager
    async def acquire_resources(self, operation: str = "unknown"):
        """Acquire resources for an operation.

        Parameters
        ----------
        operation : str, optional
            Operation name for logging

        Raises
        ------
        ResourceError
            If resources cannot be acquired
        """
        start_time = time.time()

        try:
            # Check memory limits
            self.memory_manager.check_memory_limit()

            # Acquire request slot
            async with self.request_throttler.acquire_request_slot():
                self.logger.debug(f"Resources acquired for: {operation}")
                yield

        except Exception as e:
            self.logger.error(f"Failed to acquire resources for {operation}: {e}")
            raise

        finally:
            duration = time.time() - start_time
            self.logger.debug(f"Resources released for {operation}", extra={"duration": duration})

    def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource status.

        Returns
        -------
        Dict[str, Any]
            Resource status information
        """
        try:
            return {
                "memory": self.memory_manager.get_memory_usage(),
                "throttler": self.request_throttler.get_status(),
                "system": self.monitor.get_system_stats(),
                "limits": {
                    "max_memory_mb": self.config.resources.max_memory_mb,
                    "max_concurrent": self.config.server.max_concurrent_requests,
                    "max_models": self.config.resources.max_models,
                    "max_features": self.config.resources.max_features,
                    "max_samples": self.config.resources.max_samples,
                },
            }

        except (AttributeError, ValueError, KeyError) as e:
            self.logger.error(f"Failed to get resource status: {e}")
            return {"error": str(e)}

    def check_resource_limits(self, operation_type: str, **kwargs) -> bool:
        """Check if operation is within resource limits.

        Parameters
        ----------
        operation_type : str
            Type of operation to check
        **kwargs
            Operation-specific parameters

        Returns
        -------
        bool
            True if within limits

        Raises
        ------
        ResourceError
            If limits would be exceeded
        """
        try:
            # Memory check
            self.memory_manager.check_memory_limit()

            # Operation-specific checks
            if operation_type == "train_model":
                sample_count = kwargs.get("sample_count", 0)
                feature_count = kwargs.get("feature_count", 0)

                if sample_count > self.config.resources.max_samples:
                    raise ResourceError(
                        f"Sample count exceeds limit: {sample_count} > {self.config.resources.max_samples}"
                    )

                if feature_count > self.config.resources.max_features:
                    raise ResourceError(
                        f"Feature count exceeds limit: {feature_count} > {self.config.resources.max_features}"
                    )

            return True

        except (AttributeError, ValueError, KeyError) as e:
            self.logger.error(f"Resource limit check failed for {operation_type}: {e}")
            raise


# Global resource manager instance
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance.

    Returns
    -------
    ResourceManager
        Global resource manager
    """
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


def check_memory_usage() -> Dict[str, Any]:
    """Check current memory usage.

    Returns
    -------
    Dict[str, Any]
        Memory usage information
    """
    manager = get_resource_manager()
    return manager.memory_manager.get_memory_usage()


def enforce_limits(operation_type: str):
    """Decorator to enforce resource limits on operations.

    Parameters
    ----------
    operation_type : str
        Type of operation for limit checking
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            manager = get_resource_manager()

            # Extract relevant parameters for limit checking
            limit_kwargs = {}
            if operation_type == "train_model" and len(args) >= 2:
                # Try to extract sample/feature counts from data
                try:
                    data = json.loads(args[1])  # Assuming data is second argument
                    if isinstance(data, dict):
                        limit_kwargs["sample_count"] = len(next(iter(data.values()), []))
                        limit_kwargs["feature_count"] = len(data) - 1  # Exclude target
                except (json.JSONDecodeError, KeyError, AttributeError):
                    pass

            # Check limits
            manager.check_resource_limits(operation_type, **limit_kwargs)

            # Execute with resource management
            async with manager.acquire_resources(operation_type):
                return await func(*args, **kwargs)

        return wrapper

    return decorator
