"""Test the services.base module comprehensively.

This test module provides comprehensive testing for the services base classes
to increase coverage from 0% to 60% target.
"""

import asyncio
import time
from datetime import datetime, UTC
from unittest.mock import Mock, patch, AsyncMock

import pytest

# Check if FastMCP dependencies are available
try:
    from fastmcp import FastMCP
    from src.services.base import (
        ServiceMetrics,
        MLService,
        ModelManager,
        DataProcessor,
        ValidationMixin,
        PersistenceMixin,
        MetricsMixin,
        ServiceRegistry,
        service_registry,
    )
    from src.validation import DataValidationResult
    from src.exceptions import MCPXGBoostException
    from src.config import AppConfig
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False

# Skip entire module if FastMCP not available
pytestmark = pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="FastMCP dependency not available in test environment")


class TestServiceMetrics:
    """Test ServiceMetrics class."""

    def test_service_metrics_initialization(self):
        """Test ServiceMetrics initialization."""
        metrics = ServiceMetrics()
        
        assert metrics.operation_counts == {}
        assert metrics.operation_times == {}
        assert metrics.error_counts == {}
        assert metrics.resource_usage["memory_peak"] == 0
        assert metrics.resource_usage["models_in_memory"] == 0
        assert metrics.resource_usage["total_requests"] == 0
        assert isinstance(metrics._start_time, datetime)

    def test_record_operation_success(self):
        """Test recording successful operation."""
        metrics = ServiceMetrics()
        
        metrics.record_operation("test_op", 1.5, success=True)
        
        assert metrics.operation_counts["test_op"] == 1
        assert metrics.operation_times["test_op"] == [1.5]
        assert "test_op" not in metrics.error_counts
        assert metrics.resource_usage["total_requests"] == 1

    def test_record_operation_failure(self):
        """Test recording failed operation."""
        metrics = ServiceMetrics()
        
        metrics.record_operation("test_op", 2.0, success=False)
        
        assert metrics.operation_counts["test_op"] == 1
        assert metrics.operation_times["test_op"] == [2.0]
        assert metrics.error_counts["test_op"] == 1
        assert metrics.resource_usage["total_requests"] == 1

    def test_record_operation_multiple_times(self):
        """Test recording multiple operations."""
        metrics = ServiceMetrics()
        
        # Record multiple successful operations
        for i in range(3):
            metrics.record_operation("test_op", float(i), success=True)
        
        # Record one failed operation
        metrics.record_operation("test_op", 3.0, success=False)
        
        assert metrics.operation_counts["test_op"] == 4
        assert metrics.operation_times["test_op"] == [0.0, 1.0, 2.0, 3.0]
        assert metrics.error_counts["test_op"] == 1
        assert metrics.resource_usage["total_requests"] == 4

    def test_record_operation_time_limit(self):
        """Test operation time limit (keeps only last 1000)."""
        metrics = ServiceMetrics()
        
        # Record 1100 operations
        for i in range(1100):
            metrics.record_operation("test_op", float(i), success=True)
        
        # Should only keep last 1000
        assert len(metrics.operation_times["test_op"]) == 1000
        assert metrics.operation_times["test_op"][0] == 100.0  # First kept time
        assert metrics.operation_times["test_op"][-1] == 1099.0  # Last time

    def test_get_summary(self):
        """Test get_summary method."""
        metrics = ServiceMetrics()
        
        # Record some operations
        metrics.record_operation("op1", 1.0, success=True)
        metrics.record_operation("op1", 2.0, success=True)
        metrics.record_operation("op1", 3.0, success=False)
        metrics.record_operation("op2", 0.5, success=True)
        
        summary = metrics.get_summary()
        
        assert isinstance(summary, dict)
        assert "uptime_seconds" in summary
        assert summary["operation_counts"] == {"op1": 3, "op2": 1}
        assert summary["error_counts"] == {"op1": 1}
        assert summary["resource_usage"]["total_requests"] == 4
        
        # Check average times
        assert "average_times" in summary
        assert summary["average_times"]["op1"]["avg"] == 2.0  # (1+2+3)/3
        assert summary["average_times"]["op1"]["min"] == 1.0
        assert summary["average_times"]["op1"]["max"] == 3.0
        assert summary["average_times"]["op1"]["count"] == 3
        assert summary["average_times"]["op2"]["avg"] == 0.5
        
        # Check error rate
        assert summary["error_rate"] == 0.25  # 1 error out of 4 operations

    def test_get_summary_empty(self):
        """Test get_summary with no operations."""
        metrics = ServiceMetrics()
        
        summary = metrics.get_summary()
        
        assert isinstance(summary, dict)
        assert summary["operation_counts"] == {}
        assert summary["error_counts"] == {}
        assert summary["error_rate"] == 0.0


class TestMLService:
    """Test MLService abstract base class."""

    def create_concrete_ml_service(self):
        """Create a concrete implementation of MLService for testing."""
        class ConcreteMLService(MLService):
            def __init__(self, name: str, mcp):
                super().__init__(name, mcp)
                self.setup_called = False
                self.cleanup_called = False
                self.tools_registered = False

            async def _setup_service(self):
                self.setup_called = True

            async def _cleanup_service(self):
                self.cleanup_called = True

            def _register_tools(self):
                self.tools_registered = True

            async def train_model(self, model_name: str, data: str, target_column: str, **parameters) -> str:
                return f"Trained model: {model_name}"

            async def predict(self, model_name: str, data: str) -> str:
                return f"Predictions for: {model_name}"

            async def get_model_info(self, model_name: str) -> str:
                return f"Info for: {model_name}"

            async def list_models(self) -> str:
                return "Available models: []"

        return ConcreteMLService

    @patch('src.services.base.get_config')
    def test_ml_service_initialization(self, mock_get_config):
        """Test MLService initialization."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config
        
        mock_mcp = Mock()
        ConcreteMLService = self.create_concrete_ml_service()
        
        service = ConcreteMLService("test_service", mock_mcp)
        
        assert service.name == "test_service"
        assert service.mcp == mock_mcp
        assert service.config == mock_config
        assert service.logger is not None
        assert isinstance(service.metrics, ServiceMetrics)
        assert service._initialized is False
        assert service._shutdown is False

    @patch('src.services.base.get_config')
    @pytest.mark.asyncio
    async def test_ml_service_initialize(self, mock_get_config):
        """Test MLService initialize method."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config
        
        mock_mcp = Mock()
        ConcreteMLService = self.create_concrete_ml_service()
        
        service = ConcreteMLService("test_service", mock_mcp)
        
        await service.initialize()
        
        assert service._initialized is True
        assert service.setup_called is True
        assert service.tools_registered is True

    @patch('src.services.base.get_config')
    @pytest.mark.asyncio
    async def test_ml_service_initialize_idempotent(self, mock_get_config):
        """Test MLService initialize is idempotent."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config
        
        mock_mcp = Mock()
        ConcreteMLService = self.create_concrete_ml_service()
        
        service = ConcreteMLService("test_service", mock_mcp)
        
        # Initialize twice
        await service.initialize()
        await service.initialize()
        
        assert service._initialized is True
        assert service.setup_called is True  # Should still be True, not called twice

    @patch('src.services.base.get_config')
    @pytest.mark.asyncio
    async def test_ml_service_shutdown(self, mock_get_config):
        """Test MLService shutdown method."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config
        
        mock_mcp = Mock()
        ConcreteMLService = self.create_concrete_ml_service()
        
        service = ConcreteMLService("test_service", mock_mcp)
        
        await service.shutdown()
        
        assert service._shutdown is True
        assert service.cleanup_called is True

    @patch('src.services.base.get_config')
    @pytest.mark.asyncio
    async def test_ml_service_shutdown_idempotent(self, mock_get_config):
        """Test MLService shutdown is idempotent."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config
        
        mock_mcp = Mock()
        ConcreteMLService = self.create_concrete_ml_service()
        
        service = ConcreteMLService("test_service", mock_mcp)
        
        # Shutdown twice
        await service.shutdown()
        await service.shutdown()
        
        assert service._shutdown is True
        assert service.cleanup_called is True  # Should still be True, not called twice

    @patch('src.services.base.get_config')
    @pytest.mark.asyncio
    async def test_operation_context_success(self, mock_get_config):
        """Test _operation_context for successful operation."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config
        
        mock_mcp = Mock()
        ConcreteMLService = self.create_concrete_ml_service()
        
        service = ConcreteMLService("test_service", mock_mcp)
        
        async with service._operation_context("test_op", param1="value1") as request_id:
            assert isinstance(request_id, str)
            # Operation should complete successfully
            pass
        
        # Check metrics were recorded
        assert service.metrics.operation_counts["test_op"] == 1
        assert len(service.metrics.operation_times["test_op"]) == 1
        assert "test_op" not in service.metrics.error_counts

    @patch('src.services.base.get_config')
    @pytest.mark.asyncio
    async def test_operation_context_failure(self, mock_get_config):
        """Test _operation_context for failed operation."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config
        
        mock_mcp = Mock()
        ConcreteMLService = self.create_concrete_ml_service()
        
        service = ConcreteMLService("test_service", mock_mcp)
        
        with pytest.raises(ValueError):
            async with service._operation_context("test_op") as request_id:
                assert isinstance(request_id, str)
                raise ValueError("Test error")
        
        # Check metrics were recorded as failure
        assert service.metrics.operation_counts["test_op"] == 1
        assert len(service.metrics.operation_times["test_op"]) == 1
        assert service.metrics.error_counts["test_op"] == 1

    @patch('src.services.base.get_config')
    def test_get_metrics(self, mock_get_config):
        """Test get_metrics method."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config
        
        mock_mcp = Mock()
        ConcreteMLService = self.create_concrete_ml_service()
        
        service = ConcreteMLService("test_service", mock_mcp)
        
        # Record some operations
        service.metrics.record_operation("test_op", 1.0, success=True)
        
        metrics = service.get_metrics()
        
        assert isinstance(metrics, dict)
        assert metrics["service_name"] == "test_service"
        assert metrics["initialized"] is False
        assert metrics["shutdown"] is False
        assert metrics["operation_counts"]["test_op"] == 1


class TestModelManager:
    """Test ModelManager abstract base class."""

    def test_model_manager_is_abstract(self):
        """Test that ModelManager cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ModelManager()

    def test_model_manager_concrete_implementation(self):
        """Test concrete implementation of ModelManager."""
        class ConcreteModelManager(ModelManager):
            async def save_model(self, name: str, model, metadata=None) -> bool:
                return True

            async def load_model(self, name: str):
                return f"model_{name}"

            async def delete_model(self, name: str) -> bool:
                return True

            async def list_models(self):
                return [{"name": "model1"}, {"name": "model2"}]

        manager = ConcreteModelManager()
        assert manager is not None

    @pytest.mark.asyncio
    async def test_model_manager_methods(self):
        """Test ModelManager methods."""
        class ConcreteModelManager(ModelManager):
            def __init__(self):
                self.models = {}

            async def save_model(self, name: str, model, metadata=None) -> bool:
                self.models[name] = {"model": model, "metadata": metadata}
                return True

            async def load_model(self, name: str):
                return self.models.get(name, {}).get("model")

            async def delete_model(self, name: str) -> bool:
                if name in self.models:
                    del self.models[name]
                    return True
                return False

            async def list_models(self):
                return [{"name": name, "metadata": info["metadata"]} 
                       for name, info in self.models.items()]

        manager = ConcreteModelManager()
        
        # Test save_model
        result = await manager.save_model("test_model", "model_data", {"type": "test"})
        assert result is True
        
        # Test load_model
        model = await manager.load_model("test_model")
        assert model == "model_data"
        
        # Test list_models
        models = await manager.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "test_model"
        
        # Test delete_model
        result = await manager.delete_model("test_model")
        assert result is True
        
        # Verify deletion
        models = await manager.list_models()
        assert len(models) == 0


class TestDataProcessor:
    """Test DataProcessor abstract base class."""

    def test_data_processor_is_abstract(self):
        """Test that DataProcessor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataProcessor()

    @pytest.mark.asyncio
    async def test_data_processor_concrete_implementation(self):
        """Test concrete implementation of DataProcessor."""
        class ConcreteDataProcessor(DataProcessor):
            async def validate_data(self, data, schema: str) -> DataValidationResult:
                return DataValidationResult(is_valid=True, errors=[], warnings=[])

            async def transform_data(self, data, transformations):
                return {"transformed": True, "original": data}

        processor = ConcreteDataProcessor()
        
        # Test validate_data
        result = await processor.validate_data({"test": "data"}, "test_schema")
        assert result.is_valid is True
        
        # Test transform_data
        result = await processor.transform_data({"test": "data"}, ["normalize"])
        assert result["transformed"] is True
        assert result["original"] == {"test": "data"}


class TestValidationMixin:
    """Test ValidationMixin."""

    def test_validation_mixin_methods(self):
        """Test ValidationMixin methods."""
        class TestClass(ValidationMixin):
            pass

        obj = TestClass()
        
        # Test validate_training_data
        with patch('src.services.base.validate_training_data') as mock_validate:
            mock_validate.return_value = DataValidationResult(is_valid=True, errors=[], warnings=[])
            
            result = obj.validate_training_data({"features": [1, 2, 3]}, "target")
            assert result.is_valid is True
            mock_validate.assert_called_once_with({"features": [1, 2, 3]}, "target")

        # Test validate_prediction_data
        with patch('src.services.base.validate_prediction_data') as mock_validate:
            mock_validate.return_value = DataValidationResult(is_valid=True, errors=[], warnings=[])
            
            result = obj.validate_prediction_data({"features": [1, 2, 3]}, ["f1", "f2"])
            assert result.is_valid is True
            mock_validate.assert_called_once_with({"features": [1, 2, 3]}, ["f1", "f2"])


class TestPersistenceMixin:
    """Test PersistenceMixin."""

    def test_persistence_mixin_initialization(self):
        """Test PersistenceMixin initialization."""
        class TestClass(PersistenceMixin):
            pass

        obj = TestClass()
        assert hasattr(obj, '_storage')
        assert obj._storage is not None

    def test_persistence_mixin_store_model(self):
        """Test _store_model method."""
        class TestClass(PersistenceMixin):
            pass

        obj = TestClass()
        
        # Mock the storage to avoid actual persistence
        with patch.object(obj._storage, 'save_model') as mock_save:
            obj._store_model("test_model", "model_data", {"type": "test"})
            mock_save.assert_called_once_with("test_model", "model_data", {"type": "test"})

    def test_persistence_mixin_retrieve_model(self):
        """Test _retrieve_model method."""
        class TestClass(PersistenceMixin):
            pass

        obj = TestClass()
        
        # Mock successful retrieval
        with patch.object(obj._storage, 'load_model') as mock_load:
            mock_load.return_value = ("model_data", Mock())
            
            model = obj._retrieve_model("test_model")
            assert model == "model_data"
            mock_load.assert_called_once_with("test_model")
        
        # Mock failed retrieval
        with patch.object(obj._storage, 'load_model') as mock_load:
            mock_load.side_effect = Exception("Model not found")
            
            model = obj._retrieve_model("non_existent")
            assert model is None

    def test_persistence_mixin_delete_model(self):
        """Test _delete_model method."""
        class TestClass(PersistenceMixin):
            pass

        obj = TestClass()
        
        # Mock successful deletion
        with patch.object(obj._storage, 'delete_model') as mock_delete:
            mock_delete.return_value = True
            
            result = obj._delete_model("test_model")
            assert result is True
            mock_delete.assert_called_once_with("test_model")
        
        # Mock failed deletion
        with patch.object(obj._storage, 'delete_model') as mock_delete:
            mock_delete.side_effect = Exception("Delete failed")
            
            result = obj._delete_model("non_existent")
            assert result is False

    def test_persistence_mixin_list_models(self):
        """Test _list_models method."""
        class TestClass(PersistenceMixin):
            pass

        obj = TestClass()
        
        # Create mock metadata objects
        mock_metadata1 = Mock()
        mock_metadata1.name = "model1"
        mock_metadata1.description = "First model"
        mock_metadata1.model_type = "regression"
        mock_metadata1.version = 1
        mock_metadata1.created_at = datetime.now(UTC)
        
        mock_metadata2 = Mock()
        mock_metadata2.name = "model2"
        mock_metadata2.description = "Second model"
        mock_metadata2.model_type = "classification"
        mock_metadata2.version = 2
        mock_metadata2.created_at = datetime.now(UTC)
        
        # Mock successful listing
        with patch.object(obj._storage, 'list_models') as mock_list:
            mock_list.return_value = [mock_metadata1, mock_metadata2]
            
            models = obj._list_models()
            
            assert len(models) == 2
            assert models[0]["name"] == "model1"
            assert models[0]["metadata"]["model_type"] == "regression"
            assert models[1]["name"] == "model2"
            assert models[1]["metadata"]["model_type"] == "classification"
        
        # Mock failed listing
        with patch.object(obj._storage, 'list_models') as mock_list:
            mock_list.side_effect = Exception("List failed")
            
            models = obj._list_models()
            assert models == []


class TestMetricsMixin:
    """Test MetricsMixin."""

    def test_metrics_mixin_initialization(self):
        """Test MetricsMixin initialization."""
        class TestClass(MetricsMixin):
            pass

        obj = TestClass()
        assert isinstance(obj.metrics, ServiceMetrics)

    def test_metrics_mixin_record_operation(self):
        """Test record_operation method."""
        class TestClass(MetricsMixin):
            pass

        obj = TestClass()
        
        obj.record_operation("test_op", 1.5, success=True)
        
        assert obj.metrics.operation_counts["test_op"] == 1
        assert obj.metrics.operation_times["test_op"] == [1.5]

    def test_metrics_mixin_get_metrics_summary(self):
        """Test get_metrics_summary method."""
        class TestClass(MetricsMixin):
            pass

        obj = TestClass()
        
        obj.record_operation("test_op", 1.0, success=True)
        summary = obj.get_metrics_summary()
        
        assert isinstance(summary, dict)
        assert summary["operation_counts"]["test_op"] == 1


class TestServiceRegistry:
    """Test ServiceRegistry class."""

    def test_service_registry_initialization(self):
        """Test ServiceRegistry initialization."""
        registry = ServiceRegistry()
        
        assert registry.services == {}
        assert registry.instances == {}
        assert registry.logger is not None

    def test_register_service(self):
        """Test register method."""
        registry = ServiceRegistry()
        
        # Create a mock service class
        class MockService(MLService):
            async def _setup_service(self):
                pass
            async def _cleanup_service(self):
                pass
            def _register_tools(self):
                pass
            async def train_model(self, model_name: str, data: str, target_column: str, **parameters) -> str:
                return ""
            async def predict(self, model_name: str, data: str) -> str:
                return ""
            async def get_model_info(self, model_name: str) -> str:
                return ""
            async def list_models(self) -> str:
                return ""
        
        registry.register("test_service", MockService)
        
        assert "test_service" in registry.services
        assert registry.services["test_service"] == MockService

    def test_register_invalid_service(self):
        """Test register with invalid service class."""
        registry = ServiceRegistry()
        
        class InvalidService:
            pass
        
        with pytest.raises(ValueError):
            registry.register("invalid_service", InvalidService)

    @pytest.mark.asyncio
    async def test_unregister_service(self):
        """Test unregister method."""
        registry = ServiceRegistry()
        
        # Create and register a mock service
        class MockService(MLService):
            async def _setup_service(self):
                pass
            async def _cleanup_service(self):
                pass
            def _register_tools(self):
                pass
            async def train_model(self, model_name: str, data: str, target_column: str, **parameters) -> str:
                return ""
            async def predict(self, model_name: str, data: str) -> str:
                return ""
            async def get_model_info(self, model_name: str) -> str:
                return ""
            async def list_models(self) -> str:
                return ""
        
        registry.register("test_service", MockService)
        
        # Add a mock instance
        mock_instance = Mock()
        mock_instance.shutdown = AsyncMock()
        registry.instances["test_service"] = mock_instance
        
        await registry.unregister("test_service")
        
        assert "test_service" not in registry.services
        assert "test_service" not in registry.instances

    @pytest.mark.asyncio
    async def test_get_service_new_instance(self):
        """Test get_service with new instance creation."""
        registry = ServiceRegistry()
        
        # Create a mock service class
        class MockService(MLService):
            def __init__(self, name: str, mcp):
                super().__init__(name, mcp)
                self.initialized = False
                
            async def _setup_service(self):
                pass
            async def _cleanup_service(self):
                pass
            def _register_tools(self):
                pass
            async def initialize(self):
                self.initialized = True
            async def train_model(self, model_name: str, data: str, target_column: str, **parameters) -> str:
                return ""
            async def predict(self, model_name: str, data: str) -> str:
                return ""
            async def get_model_info(self, model_name: str) -> str:
                return ""
            async def list_models(self) -> str:
                return ""
        
        registry.register("test_service", MockService)
        
        mock_mcp = Mock()
        service = await registry.get_service("test_service", mock_mcp)
        
        assert isinstance(service, MockService)
        assert service.initialized is True
        assert "test_service" in registry.instances

    @pytest.mark.asyncio
    async def test_get_service_existing_instance(self):
        """Test get_service with existing instance."""
        registry = ServiceRegistry()
        
        # Create a mock service class
        class MockService(MLService):
            async def _setup_service(self):
                pass
            async def _cleanup_service(self):
                pass
            def _register_tools(self):
                pass
            async def train_model(self, model_name: str, data: str, target_column: str, **parameters) -> str:
                return ""
            async def predict(self, model_name: str, data: str) -> str:
                return ""
            async def get_model_info(self, model_name: str) -> str:
                return ""
            async def list_models(self) -> str:
                return ""
        
        registry.register("test_service", MockService)
        
        # Add existing instance
        existing_instance = Mock()
        registry.instances["test_service"] = existing_instance
        
        mock_mcp = Mock()
        service = await registry.get_service("test_service", mock_mcp)
        
        assert service == existing_instance

    @pytest.mark.asyncio
    async def test_get_service_not_registered(self):
        """Test get_service with unregistered service."""
        registry = ServiceRegistry()
        
        mock_mcp = Mock()
        
        with pytest.raises(ValueError):
            await registry.get_service("unregistered_service", mock_mcp)

    def test_list_services(self):
        """Test list_services method."""
        registry = ServiceRegistry()
        
        # Create mock service classes
        class MockService1(MLService):
            async def _setup_service(self):
                pass
            async def _cleanup_service(self):
                pass
            def _register_tools(self):
                pass
            async def train_model(self, model_name: str, data: str, target_column: str, **parameters) -> str:
                return ""
            async def predict(self, model_name: str, data: str) -> str:
                return ""
            async def get_model_info(self, model_name: str) -> str:
                return ""
            async def list_models(self) -> str:
                return ""
        
        class MockService2(MLService):
            async def _setup_service(self):
                pass
            async def _cleanup_service(self):
                pass
            def _register_tools(self):
                pass
            async def train_model(self, model_name: str, data: str, target_column: str, **parameters) -> str:
                return ""
            async def predict(self, model_name: str, data: str) -> str:
                return ""
            async def get_model_info(self, model_name: str) -> str:
                return ""
            async def list_models(self) -> str:
                return ""
        
        registry.register("service1", MockService1)
        registry.register("service2", MockService2)
        
        services = registry.list_services()
        
        assert len(services) == 2
        assert "service1" in services
        assert "service2" in services

    def test_list_instances(self):
        """Test list_instances method."""
        registry = ServiceRegistry()
        
        # Add mock instances
        mock_instance1 = Mock()
        mock_instance2 = Mock()
        registry.instances["service1"] = mock_instance1
        registry.instances["service2"] = mock_instance2
        
        instances = registry.list_instances()
        
        assert len(instances) == 2
        assert "service1" in instances
        assert "service2" in instances

    @pytest.mark.asyncio
    async def test_shutdown_all(self):
        """Test shutdown_all method."""
        registry = ServiceRegistry()
        
        # Add mock instances
        mock_instance1 = Mock()
        mock_instance1.shutdown = AsyncMock()
        mock_instance2 = Mock()
        mock_instance2.shutdown = AsyncMock()
        
        registry.instances["service1"] = mock_instance1
        registry.instances["service2"] = mock_instance2
        
        await registry.shutdown_all()
        
        mock_instance1.shutdown.assert_called_once()
        mock_instance2.shutdown.assert_called_once()
        assert len(registry.instances) == 0


class TestGlobalServiceRegistry:
    """Test global service registry."""

    def test_global_service_registry_exists(self):
        """Test that global service registry exists."""
        assert service_registry is not None
        assert isinstance(service_registry, ServiceRegistry) 