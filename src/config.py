"""Configuration Management for MCP XGBoost Server.

This module provides centralized configuration management for the MCP XGBoost
server application. It handles environment variables, default values, and
configuration validation using Pydantic models.

The configuration is organized into logical sections:
- Server configuration (host, port, etc.)
- XGBoost model defaults
- Resource limits and constraints
- Logging and monitoring settings
- Storage and persistence options

Classes
-------
ServerConfig
    Server-related configuration settings
XGBoostConfig
    XGBoost model configuration and defaults
ResourceConfig
    Resource management and limits configuration
LoggingConfig
    Logging and monitoring configuration
StorageConfig
    Data storage and model persistence configuration
RedisConfig
    Redis configuration for distributed model storage
AppConfig
    Main application configuration container

Functions
---------
get_config
    Get the global configuration instance
load_config
    Load configuration from environment variables

Notes
-----
Configuration values are loaded from environment variables with sensible
defaults. The configuration is validated using Pydantic models to ensure
type safety and proper values.

Environment Variables
--------------------
MCP_HOST : str
    Server host address (default: "127.0.0.1")
MCP_PORT : int
    Server port number (default: 8000)
MCP_DEBUG : bool
    Enable debug mode (default: False)
MCP_LOG_LEVEL : str
    Logging level (default: "INFO")
MCP_MODEL_STORAGE_PATH : str
    Path for model persistence (default: "./models")
MCP_MAX_MODELS : int
    Maximum number of models to keep in memory (default: 10)
MCP_MAX_FEATURES : int
    Maximum number of features per dataset (default: 1000)
MCP_MAX_SAMPLES : int
    Maximum number of samples per dataset (default: 100000)
MCP_REQUEST_TIMEOUT : int
    Request timeout in seconds (default: 300)

Examples
--------
Load and use configuration:

    >>> from src.config import get_config
    >>> config = get_config()
    >>> print(config.server.host)
    '127.0.0.1'
    >>> print(config.xgboost.n_estimators)
    100

See Also
--------
pydantic : Data validation and settings management
src.constants : Legacy constants (to be migrated)
"""

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ServerConfig(BaseModel):
    """Server configuration settings.

    Attributes
    ----------
    host : str
        Server host address
    port : int
        Server port number
    debug : bool
        Enable debug mode
    request_timeout : int
        Request timeout in seconds
    max_concurrent_requests : int
        Maximum number of concurrent requests
    """

    host: str = Field(default="127.0.0.1", description="Server host address")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port number")
    debug: bool = Field(default=False, description="Enable debug mode")
    request_timeout: int = Field(default=300, ge=1, description="Request timeout in seconds")
    max_concurrent_requests: int = Field(default=10, ge=1, description="Maximum concurrent requests")

    @field_validator("host")
    @classmethod
    def validate_host(cls, v):
        """Validate host address format."""
        if not v or not isinstance(v, str):
            raise ValueError("Host must be a non-empty string")
        return v


class XGBoostConfig(BaseModel):
    """XGBoost model configuration and defaults.

    Attributes
    ----------
    n_estimators : int
        Default number of estimators
    max_depth : int
        Default maximum tree depth
    learning_rate : float
        Default learning rate
    subsample : float
        Default subsample ratio
    colsample_bytree : float
        Default column sample ratio
    random_state : int
        Default random seed
    """

    n_estimators: int = Field(default=100, ge=1, description="Default number of estimators")
    max_depth: int = Field(default=6, ge=1, le=20, description="Default maximum tree depth")
    learning_rate: float = Field(default=0.3, gt=0, le=1, description="Default learning rate")
    subsample: float = Field(default=1.0, gt=0, le=1, description="Default subsample ratio")
    colsample_bytree: float = Field(default=1.0, gt=0, le=1, description="Default column sample ratio")
    random_state: int = Field(default=42, ge=0, description="Default random seed")

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for XGBoost parameters."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
        }


class ResourceConfig(BaseModel):
    """Resource management and limits configuration.

    Attributes
    ----------
    max_models : int
        Maximum number of models to keep in memory
    max_features : int
        Maximum number of features per dataset
    max_samples : int
        Maximum number of samples per dataset
    min_samples : int
        Minimum number of samples for training
    max_memory_mb : int
        Maximum memory usage in MB
    """

    max_models: int = Field(default=10, ge=1, description="Maximum models in memory")
    max_features: int = Field(default=1000, ge=1, description="Maximum features per dataset")
    max_samples: int = Field(default=100000, ge=1, description="Maximum samples per dataset")
    min_samples: int = Field(default=1, ge=1, description="Minimum samples for training")
    max_memory_mb: int = Field(default=1024, ge=128, description="Maximum memory usage in MB")

    @field_validator("min_samples")
    @classmethod
    def validate_min_samples(cls, v, info):
        """Ensure min_samples is reasonable compared to max_samples."""
        if info.data and "max_samples" in info.data:
            max_samples = info.data["max_samples"]
            if v >= max_samples:
                raise ValueError("min_samples must be less than max_samples")
        return v


class LoggingConfig(BaseModel):
    """Logging and monitoring configuration.

    Attributes
    ----------
    level : str
        Logging level
    format : str
        Log message format
    file_path : Optional[str]
        Path to log file (None for console only)
    max_file_size_mb : int
        Maximum log file size in MB
    backup_count : int
        Number of backup log files to keep
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log message format"
    )
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size_mb: int = Field(default=10, ge=1, description="Maximum log file size in MB")
    backup_count: int = Field(default=5, ge=1, description="Number of backup log files")


class StorageConfig(BaseModel):
    """Data storage and model persistence configuration.

    Attributes
    ----------
    model_storage_path : str
        Path for model persistence
    data_storage_path : str
        Path for data storage
    enable_model_persistence : bool
        Enable model persistence to disk
    auto_save_models : bool
        Automatically save models after training
    compression : bool
        Enable compression for stored models
    storage_backend : str
        Storage backend type ('file' or 'redis')
    """

    model_storage_path: str = Field(default="./models", description="Model storage path")
    data_storage_path: str = Field(default="./data", description="Data storage path")
    enable_model_persistence: bool = Field(default=True, description="Enable model persistence")
    auto_save_models: bool = Field(default=True, description="Auto-save models after training")
    compression: bool = Field(default=True, description="Enable model compression")
    storage_backend: str = Field(default="file", description="Storage backend type ('file' or 'redis')")

    @field_validator("model_storage_path", "data_storage_path")
    @classmethod
    def validate_paths(cls, v):
        """Validate and create storage paths."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise ValueError(f"Cannot create or access directory: {v}")
        return str(path.absolute())

    @field_validator("storage_backend")
    @classmethod
    def validate_storage_backend(cls, v):
        """Validate storage backend type."""
        if v not in ["file", "redis"]:
            raise ValueError("Storage backend must be 'file' or 'redis'")
        return v


class RedisConfig(BaseModel):
    """Redis configuration for distributed model storage.

    Attributes
    ----------
    host : str
        Redis host address
    port : int
        Redis port number
    password : Optional[str]
        Redis password
    database : int
        Redis database number
    max_connections : int
        Maximum number of connections in the pool
    socket_timeout : int
        Socket timeout in seconds
    socket_connect_timeout : int
        Socket connect timeout in seconds
    retry_on_timeout : bool
        Retry on timeout
    health_check_interval : int
        Health check interval in seconds
    """

    host: str = Field(default="localhost", description="Redis host address")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port number")
    password: Optional[str] = Field(default=None, description="Redis password")
    database: int = Field(default=0, ge=0, le=15, description="Redis database number")
    max_connections: int = Field(default=10, ge=1, description="Maximum connections in pool")
    socket_timeout: int = Field(default=5, ge=1, description="Socket timeout in seconds")
    socket_connect_timeout: int = Field(default=5, ge=1, description="Socket connect timeout in seconds")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    health_check_interval: int = Field(default=30, ge=1, description="Health check interval in seconds")

    @field_validator("host")
    @classmethod
    def validate_host(cls, v):
        """Validate Redis host address."""
        if not v or not v.strip():
            raise ValueError("Redis host cannot be empty")
        return v.strip()


class AppConfig(BaseModel):
    """Main application configuration container.

    Attributes
    ----------
    server : ServerConfig
        Server configuration
    xgboost : XGBoostConfig
        XGBoost configuration
    resources : ResourceConfig
        Resource management configuration
    logging : LoggingConfig
        Logging configuration
    storage : StorageConfig
        Storage configuration
    redis : RedisConfig
        Redis configuration
    app_name : str
        Application name
    version : str
        Application version
    """

    server: ServerConfig = Field(default_factory=ServerConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    app_name: str = Field(default="mcp-xgboost", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")

    class Config:
        """Pydantic model configuration."""

        env_prefix = "MCP_"
        case_sensitive = False


def load_config() -> AppConfig:
    """Load configuration from environment variables.

    Returns
    -------
    AppConfig
        Configured application settings

    Notes
    -----
    This function loads configuration from environment variables with the
    prefix "MCP_". It creates the necessary directories and validates all
    configuration values.

    Environment variables are mapped to configuration fields using the
    following pattern:
    - MCP_HOST -> server.host
    - MCP_PORT -> server.port
    - MCP_DEBUG -> server.debug
    - MCP_LOG_LEVEL -> logging.level
    - MCP_MODEL_STORAGE_PATH -> storage.model_storage_path
    - etc.
    """
    # Create base configuration
    config = AppConfig()

    # Load server configuration
    config.server.host = os.getenv("MCP_HOST", config.server.host)
    config.server.port = int(os.getenv("MCP_PORT", config.server.port))
    config.server.debug = os.getenv("MCP_DEBUG", "false").lower() == "true"
    config.server.request_timeout = int(os.getenv("MCP_REQUEST_TIMEOUT", config.server.request_timeout))
    config.server.max_concurrent_requests = int(
        os.getenv("MCP_MAX_CONCURRENT_REQUESTS", config.server.max_concurrent_requests)
    )

    # Load XGBoost configuration
    config.xgboost.n_estimators = int(os.getenv("MCP_N_ESTIMATORS", config.xgboost.n_estimators))
    config.xgboost.max_depth = int(os.getenv("MCP_MAX_DEPTH", config.xgboost.max_depth))
    config.xgboost.learning_rate = float(os.getenv("MCP_LEARNING_RATE", config.xgboost.learning_rate))
    config.xgboost.subsample = float(os.getenv("MCP_SUBSAMPLE", config.xgboost.subsample))
    config.xgboost.colsample_bytree = float(os.getenv("MCP_COLSAMPLE_BYTREE", config.xgboost.colsample_bytree))
    config.xgboost.random_state = int(os.getenv("MCP_RANDOM_STATE", config.xgboost.random_state))

    # Load resource configuration
    config.resources.max_models = int(os.getenv("MCP_MAX_MODELS", config.resources.max_models))
    config.resources.max_features = int(os.getenv("MCP_MAX_FEATURES", config.resources.max_features))
    config.resources.max_samples = int(os.getenv("MCP_MAX_SAMPLES", config.resources.max_samples))
    config.resources.min_samples = int(os.getenv("MCP_MIN_SAMPLES", config.resources.min_samples))
    config.resources.max_memory_mb = int(os.getenv("MCP_MAX_MEMORY_MB", config.resources.max_memory_mb))

    # Load logging configuration
    log_level = os.getenv("MCP_LOG_LEVEL", config.logging.level).upper()
    if log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        config.logging.level = log_level
    config.logging.file_path = os.getenv("MCP_LOG_FILE", config.logging.file_path)
    config.logging.max_file_size_mb = int(os.getenv("MCP_LOG_MAX_SIZE_MB", config.logging.max_file_size_mb))
    config.logging.backup_count = int(os.getenv("MCP_LOG_BACKUP_COUNT", config.logging.backup_count))

    # Load storage configuration
    config.storage.model_storage_path = os.getenv("MCP_MODEL_STORAGE_PATH", config.storage.model_storage_path)
    config.storage.data_storage_path = os.getenv("MCP_DATA_STORAGE_PATH", config.storage.data_storage_path)
    config.storage.enable_model_persistence = os.getenv("MCP_ENABLE_MODEL_PERSISTENCE", "true").lower() == "true"
    config.storage.auto_save_models = os.getenv("MCP_AUTO_SAVE_MODELS", "true").lower() == "true"
    config.storage.compression = os.getenv("MCP_COMPRESSION", "true").lower() == "true"
    config.storage.storage_backend = os.getenv("MCP_STORAGE_BACKEND", config.storage.storage_backend)

    # Load Redis configuration
    config.redis.host = os.getenv("MCP_REDIS_HOST", config.redis.host)
    config.redis.port = int(os.getenv("MCP_REDIS_PORT", config.redis.port))
    config.redis.password = os.getenv("MCP_REDIS_PASSWORD", config.redis.password)
    config.redis.database = int(os.getenv("MCP_REDIS_DATABASE", config.redis.database))
    config.redis.max_connections = int(os.getenv("MCP_REDIS_MAX_CONNECTIONS", config.redis.max_connections))
    config.redis.socket_timeout = int(os.getenv("MCP_REDIS_SOCKET_TIMEOUT", config.redis.socket_timeout))
    config.redis.socket_connect_timeout = int(os.getenv("MCP_REDIS_SOCKET_CONNECT_TIMEOUT", config.redis.socket_connect_timeout))
    config.redis.retry_on_timeout = os.getenv("MCP_REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
    config.redis.health_check_interval = int(os.getenv("MCP_REDIS_HEALTH_CHECK_INTERVAL", config.redis.health_check_interval))

    # Load application metadata
    config.app_name = os.getenv("MCP_APP_NAME", config.app_name)
    config.version = os.getenv("MCP_VERSION", config.version)

    return config


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance.

    Returns
    -------
    AppConfig
        Global configuration instance

    Notes
    -----
    This function implements a singleton pattern for configuration access.
    The configuration is loaded once and reused throughout the application.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> AppConfig:
    """Reload configuration from environment variables.

    Returns
    -------
    AppConfig
        Reloaded configuration instance

    Notes
    -----
    This function forces a reload of the configuration, useful for testing
    or when configuration changes need to be applied at runtime.
    """
    global _config
    _config = load_config()
    return _config
