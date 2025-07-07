"""Tests for the configuration module."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from src.config import AppConfig, ServerConfig, XGBoostConfig, ResourceConfig, LoggingConfig, StorageConfig


class TestAppConfig:
    """Test the AppConfig class."""

    def test_default_config_creation(self):
        """Test creating a config with default values."""
        config = AppConfig()
        
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.xgboost, XGBoostConfig)
        assert isinstance(config.resources, ResourceConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.storage, StorageConfig)
        
        # Check default values
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 8000
        assert config.xgboost.max_depth == 6
        assert config.resources.max_memory_mb == 1024

    def test_config_from_environment_variables(self):
        """Test config loading from environment variables."""
        env_vars = {
            'MCP_HOST': '0.0.0.0',
            'MCP_PORT': '9000',
            'MCP_MAX_DEPTH': '8',
            'MCP_MAX_MEMORY_MB': '2048',
            'MCP_LOG_LEVEL': 'DEBUG'
        }
        
        with patch.dict(os.environ, env_vars):
            from src.config import reload_config
            config = reload_config()
            
            assert config.server.host == "0.0.0.0"
            assert config.server.port == 9000
            assert config.xgboost.max_depth == 8
            assert config.resources.max_memory_mb == 2048
            assert config.logging.level == "DEBUG"

    def test_config_validation(self):
        """Test config validation."""
        # Test that port must be positive
        with pytest.raises(Exception):
            AppConfig(server=ServerConfig(port=-1))
            
        # Test that memory must be positive  
        with pytest.raises(Exception):
            AppConfig(resources=ResourceConfig(max_memory_mb=-100))

    def test_config_from_function(self):
        """Test that get_config function works."""
        # Clear any environment variables from previous tests
        with patch.dict(os.environ, {}, clear=True):
            from src.config import reload_config
            config = reload_config()
            
            assert isinstance(config, AppConfig)
            assert config.server.host == "127.0.0.1"


class TestServerConfig:
    """Test the ServerConfig class."""

    def test_server_config_defaults(self):
        """Test server config default values."""
        config = ServerConfig()
        
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.debug is False

    def test_server_config_validation(self):
        """Test server config validation."""
        config = ServerConfig()
        
        # Valid port range
        config.port = 8080
        assert config.port == 8080
        
        # Test boolean values
        config.debug = True
        assert config.debug is True


class TestXGBoostConfig:
    """Test the XGBoostConfig class."""

    def test_xgboost_config_defaults(self):
        """Test XGBoost config default values."""
        config = XGBoostConfig()
        
        assert config.max_depth == 6
        assert config.learning_rate == 0.3
        assert config.n_estimators == 100
        assert config.random_state == 42

    def test_xgboost_config_validation(self):
        """Test XGBoost config validation."""
        config = XGBoostConfig()
        
        # Test positive values
        config.max_depth = 10
        assert config.max_depth == 10
        
        config.learning_rate = 0.1
        assert config.learning_rate == 0.1


class TestResourceConfig:
    """Test the ResourceConfig class."""

    def test_resource_config_defaults(self):
        """Test resource config default values."""
        config = ResourceConfig()
        
        assert config.max_memory_mb == 1024
        assert config.max_models == 10
        assert config.max_features == 1000

    def test_resource_config_validation(self):
        """Test resource config validation."""
        config = ResourceConfig()
        
        # Test positive values
        config.max_memory_mb = 2048
        assert config.max_memory_mb == 2048
        
        config.max_models = 5
        assert config.max_models == 5


class TestLoggingConfig:
    """Test the LoggingConfig class."""

    def test_logging_config_defaults(self):
        """Test logging config default values."""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert "asctime" in config.format
        assert config.file_path is None

    def test_logging_config_validation(self):
        """Test logging config validation."""
        config = LoggingConfig()
        
        # Test valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config.level = level
            assert config.level == level

        # Test format string
        config.format = "test format"
        assert config.format == "test format"


class TestStorageConfig:
    """Test the StorageConfig class."""

    def test_storage_config_defaults(self):
        """Test storage config default values."""
        config = StorageConfig()
        
        assert config.model_storage_path == "./models"
        assert config.data_storage_path == "./data"
        assert config.enable_model_persistence is True
        assert config.auto_save_models is True

    def test_storage_config_paths(self):
        """Test storage config path creation."""
        config = StorageConfig()
        config.model_storage_path = "/tmp/test/models"
        config.data_storage_path = "/tmp/test/data"
        
        # The config should provide access to these paths
        assert config.model_storage_path == "/tmp/test/models"
        assert config.data_storage_path == "/tmp/test/data" 