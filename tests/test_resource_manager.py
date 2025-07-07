"""Test the resource_manager module comprehensively.

This test module provides comprehensive testing for the resource management
to increase coverage from 48% to 75% target.
"""

import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from contextlib import asynccontextmanager

import pytest
import psutil

from src.resource_manager import (
    MemoryManager,
    RequestThrottler,
    ResourceMonitor,
    ResourceManager,
    get_resource_manager,
    check_memory_usage,
    enforce_limits,
)
from src.exceptions import MemoryLimitError, ConcurrencyLimitError, ResourceError
from src.config import AppConfig


class AsyncContextManagerMock:
    """Mock async context manager for testing."""
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TestMemoryManager:
    """Test MemoryManager class."""

    def test_memory_manager_initialization(self):
        """Test MemoryManager initialization."""
        manager = MemoryManager(max_memory_mb=1024)
        
        assert manager.max_memory_mb == 1024
        assert manager.logger is not None

    @patch('src.resource_manager.psutil.Process')
    def test_get_memory_usage(self, mock_process_class):
        """Test get_memory_usage method."""
        # Mock process and memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 512 * 1024 * 1024  # 512 MB
        mock_memory_info.vms = 1024 * 1024 * 1024  # 1024 MB
        
        mock_process = Mock()
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 25.0
        
        mock_process_class.return_value = mock_process
        
        manager = MemoryManager(max_memory_mb=1024)
        usage = manager.get_memory_usage()
        
        assert isinstance(usage, dict)
        assert usage["rss_mb"] == 512.0
        assert usage["vms_mb"] == 1024.0
        assert usage["percent"] == 25.0
        assert usage["available_mb"] == 512.0  # 1024 - 512
        assert usage["limit_mb"] == 1024

    @patch('src.resource_manager.psutil.Process')
    def test_check_memory_limit_within_bounds(self, mock_process_class):
        """Test check_memory_limit when within bounds."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 512 * 1024 * 1024  # 512 MB
        mock_memory_info.vms = 1024 * 1024 * 1024  # 1024 MB
        
        mock_process = Mock()
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 25.0
        
        mock_process_class.return_value = mock_process
        
        manager = MemoryManager(max_memory_mb=1024)
        result = manager.check_memory_limit()
        
        assert result is True

    @patch('src.resource_manager.psutil.Process')
    def test_check_memory_limit_exceeds_bounds(self, mock_process_class):
        """Test check_memory_limit when exceeding bounds."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 1536 * 1024 * 1024  # 1536 MB (exceeds limit)
        mock_memory_info.vms = 2048 * 1024 * 1024  # 2048 MB
        
        mock_process = Mock()
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 75.0
        
        mock_process_class.return_value = mock_process
        
        manager = MemoryManager(max_memory_mb=1024)
        
        with pytest.raises(MemoryLimitError):
            manager.check_memory_limit()


class TestRequestThrottler:
    """Test RequestThrottler class."""

    def test_request_throttler_initialization(self):
        """Test RequestThrottler initialization."""
        throttler = RequestThrottler(max_concurrent=10, rate_limit=60)
        
        assert throttler.max_concurrent == 10
        assert throttler.rate_limit == 60
        assert throttler.current_requests == 0
        assert throttler.request_times == []
        assert throttler.semaphore._value == 10

    @pytest.mark.asyncio
    async def test_acquire_request_slot_success(self):
        """Test successful request slot acquisition."""
        throttler = RequestThrottler(max_concurrent=2, rate_limit=10)
        
        async with throttler.acquire_request_slot():
            assert throttler.current_requests == 1
            assert len(throttler.request_times) == 1
        
        assert throttler.current_requests == 0

    @pytest.mark.asyncio
    async def test_acquire_request_slot_concurrency_limit(self):
        """Test request slot acquisition with concurrency limit."""
        throttler = RequestThrottler(max_concurrent=1, rate_limit=10)
        
        # First request should succeed
        async with throttler.acquire_request_slot():
            assert throttler.current_requests == 1
            
            # Second request should wait
            async def second_request():
                async with throttler.acquire_request_slot():
                    assert throttler.current_requests == 1
            
            # Start second request but don't wait for it
            task = asyncio.create_task(second_request())
            await asyncio.sleep(0.1)  # Give it time to start
            
            # Should still be 1 request
            assert throttler.current_requests == 1
            
            # Cancel the task to avoid hanging
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_acquire_request_slot_rate_limit(self):
        """Test request slot acquisition with rate limit exceeded."""
        throttler = RequestThrottler(max_concurrent=10, rate_limit=2)
        
        # Fill up the rate limit
        for _ in range(2):
            async with throttler.acquire_request_slot():
                pass
        
        # Next request should fail
        with pytest.raises(ConcurrencyLimitError):
            async with throttler.acquire_request_slot():
                pass

    def test_get_status(self):
        """Test get_status method."""
        throttler = RequestThrottler(max_concurrent=5, rate_limit=30)
        
        # Add some request times
        now = time.time()
        throttler.request_times = [now - 30, now - 20, now - 10]
        throttler.current_requests = 2
        
        status = throttler.get_status()
        
        assert isinstance(status, dict)
        assert status["current_requests"] == 2
        assert status["max_concurrent"] == 5
        assert status["requests_last_minute"] == 3
        assert status["rate_limit"] == 30
        assert status["available_slots"] == 3

    def test_get_status_filters_old_requests(self):
        """Test get_status filters out old requests."""
        throttler = RequestThrottler(max_concurrent=5, rate_limit=30)
        
        # Add request times (some older than 1 minute)
        now = time.time()
        throttler.request_times = [
            now - 120,  # 2 minutes ago (should be filtered)
            now - 30,   # 30 seconds ago
            now - 10    # 10 seconds ago
        ]
        
        status = throttler.get_status()
        
        assert status["requests_last_minute"] == 2  # Only recent requests counted


class TestResourceMonitor:
    """Test ResourceMonitor class."""

    def test_resource_monitor_initialization(self):
        """Test ResourceMonitor initialization."""
        monitor = ResourceMonitor()
        
        assert monitor.logger is not None
        assert isinstance(monitor.start_time, float)

    @patch('src.resource_manager.psutil.cpu_percent')
    @patch('src.resource_manager.psutil.cpu_count')
    @patch('src.resource_manager.psutil.virtual_memory')
    @patch('src.resource_manager.psutil.disk_usage')
    @patch('src.resource_manager.psutil.Process')
    @patch('src.resource_manager.psutil.getloadavg')
    def test_get_system_stats_success(self, mock_getloadavg, mock_process_class,
                                      mock_disk_usage, mock_virtual_memory,
                                      mock_cpu_count, mock_cpu_percent):
        """Test successful get_system_stats method."""
        # Mock CPU info
        mock_cpu_percent.return_value = 25.5
        mock_cpu_count.return_value = 4
        mock_getloadavg.return_value = (1.0, 1.5, 2.0)
        
        # Mock memory info
        mock_memory = Mock()
        mock_memory.total = 8 * 1024 * 1024 * 1024  # 8 GB
        mock_memory.available = 4 * 1024 * 1024 * 1024  # 4 GB
        mock_memory.used = 4 * 1024 * 1024 * 1024  # 4 GB
        mock_memory.percent = 50.0
        mock_virtual_memory.return_value = mock_memory
        
        # Mock disk info
        mock_disk = Mock()
        mock_disk.total = 1024 * 1024 * 1024 * 1024  # 1 TB
        mock_disk.free = 512 * 1024 * 1024 * 1024  # 512 GB
        mock_disk.used = 512 * 1024 * 1024 * 1024  # 512 GB
        mock_disk_usage.return_value = mock_disk
        
        # Mock process info
        mock_process_memory = Mock()
        mock_process_memory.rss = 256 * 1024 * 1024  # 256 MB
        
        mock_process = Mock()
        mock_process.memory_info.return_value = mock_process_memory
        mock_process.memory_percent.return_value = 3.125
        mock_process.cpu_percent.return_value = 15.0
        mock_process.num_threads.return_value = 8
        mock_process_class.return_value = mock_process
        
        monitor = ResourceMonitor()
        stats = monitor.get_system_stats()
        
        assert isinstance(stats, dict)
        assert stats["cpu"]["percent"] == 25.5
        assert stats["cpu"]["count"] == 4
        assert stats["cpu"]["load_avg"] == (1.0, 1.5, 2.0)
        assert stats["memory"]["total_mb"] == 8192.0
        assert stats["memory"]["available_mb"] == 4096.0
        assert stats["memory"]["used_mb"] == 4096.0
        assert stats["memory"]["percent"] == 50.0
        assert stats["disk"]["total_gb"] == 1024.0
        assert stats["disk"]["free_gb"] == 512.0
        assert stats["disk"]["used_gb"] == 512.0
        assert stats["disk"]["percent"] == 50.0
        assert stats["process"]["memory_mb"] == 256.0
        assert stats["process"]["memory_percent"] == 3.125
        assert stats["process"]["cpu_percent"] == 15.0
        assert stats["process"]["num_threads"] == 8
        assert "uptime_seconds" in stats

    @patch('src.resource_manager.psutil.cpu_percent')
    def test_get_system_stats_error_handling(self, mock_cpu_percent):
        """Test get_system_stats error handling."""
        mock_cpu_percent.side_effect = OSError("Test error")
        
        monitor = ResourceMonitor()
        stats = monitor.get_system_stats()
        
        assert isinstance(stats, dict)
        assert "error" in stats
        assert stats["error"] == "Test error"

    @patch('src.resource_manager.psutil.getloadavg')
    def test_get_system_stats_no_loadavg(self, mock_getloadavg):
        """Test get_system_stats when getloadavg is not available."""
        # Simulate getloadavg not being available
        mock_getloadavg.side_effect = AttributeError("getloadavg not available")
        
        with patch('src.resource_manager.hasattr', return_value=False):
            monitor = ResourceMonitor()
            
            with patch('src.resource_manager.psutil.cpu_percent', return_value=10.0):
                with patch('src.resource_manager.psutil.cpu_count', return_value=2):
                    with patch('src.resource_manager.psutil.virtual_memory'):
                        with patch('src.resource_manager.psutil.disk_usage'):
                            with patch('src.resource_manager.psutil.Process'):
                                stats = monitor.get_system_stats()
                                
                                assert stats["cpu"]["load_avg"] is None


class TestResourceManager:
    """Test ResourceManager class."""

    def setup_method(self):
        """Setup test environment."""
        self.mock_config = Mock()
        self.mock_config.resources.max_memory_mb = 1024
        self.mock_config.resources.max_models = 10
        self.mock_config.resources.max_features = 1000
        self.mock_config.resources.max_samples = 10000
        self.mock_config.server.max_concurrent_requests = 5

    def test_resource_manager_initialization(self):
        """Test ResourceManager initialization."""
        manager = ResourceManager(self.mock_config)
        
        assert manager.config == self.mock_config
        assert manager.logger is not None
        assert isinstance(manager.memory_manager, MemoryManager)
        assert isinstance(manager.request_throttler, RequestThrottler)
        assert isinstance(manager.monitor, ResourceMonitor)

    @patch('src.resource_manager.get_config')
    def test_resource_manager_default_config(self, mock_get_config):
        """Test ResourceManager with default config."""
        mock_get_config.return_value = self.mock_config
        
        manager = ResourceManager()
        
        assert manager.config == self.mock_config

    @pytest.mark.asyncio
    async def test_acquire_resources_success(self):
        """Test successful resource acquisition."""
        manager = ResourceManager(self.mock_config)
        
        # Mock memory check to succeed
        manager.memory_manager.check_memory_limit = Mock(return_value=True)
        
        async with manager.acquire_resources("test_operation"):
            # Should reach here successfully
            pass

    @pytest.mark.asyncio
    async def test_acquire_resources_memory_limit_exceeded(self):
        """Test resource acquisition with memory limit exceeded."""
        manager = ResourceManager(self.mock_config)
        
        # Mock memory check to fail
        manager.memory_manager.check_memory_limit = Mock(
            side_effect=MemoryLimitError("Memory limit exceeded", details={})
        )
        
        with pytest.raises(MemoryLimitError):
            async with manager.acquire_resources("test_operation"):
                pass

    @pytest.mark.asyncio
    async def test_acquire_resources_concurrency_limit_exceeded(self):
        """Test resource acquisition with concurrency limit exceeded."""
        manager = ResourceManager(self.mock_config)
        
        # Mock memory check to succeed
        manager.memory_manager.check_memory_limit = Mock(return_value=True)
        
        # Mock request throttler to fail
        @asynccontextmanager
        async def mock_acquire_request_slot():
            raise ConcurrencyLimitError("Too many requests", details={})
            yield  # This won't be reached
        
        manager.request_throttler.acquire_request_slot = mock_acquire_request_slot
        
        with pytest.raises(ConcurrencyLimitError):
            async with manager.acquire_resources("test_operation"):
                pass

    def test_get_resource_status_success(self):
        """Test successful get_resource_status method."""
        manager = ResourceManager(self.mock_config)
        
        # Mock all the methods
        manager.memory_manager.get_memory_usage = Mock(return_value={"rss_mb": 512})
        manager.request_throttler.get_status = Mock(return_value={"current_requests": 2})
        manager.monitor.get_system_stats = Mock(return_value={"cpu": {"percent": 25}})
        
        status = manager.get_resource_status()
        
        assert isinstance(status, dict)
        assert status["memory"] == {"rss_mb": 512}
        assert status["throttler"] == {"current_requests": 2}
        assert status["system"] == {"cpu": {"percent": 25}}
        assert status["limits"]["max_memory_mb"] == 1024
        assert status["limits"]["max_concurrent"] == 5
        assert status["limits"]["max_models"] == 10
        assert status["limits"]["max_features"] == 1000
        assert status["limits"]["max_samples"] == 10000

    def test_get_resource_status_error_handling(self):
        """Test get_resource_status error handling."""
        manager = ResourceManager(self.mock_config)
        
        # Mock methods to raise exceptions
        manager.memory_manager.get_memory_usage = Mock(side_effect=AttributeError("Test error"))
        
        status = manager.get_resource_status()
        
        assert isinstance(status, dict)
        assert "error" in status
        assert status["error"] == "Test error"

    def test_check_resource_limits_success(self):
        """Test successful resource limit checking."""
        manager = ResourceManager(self.mock_config)
        
        # Mock memory check to succeed
        manager.memory_manager.check_memory_limit = Mock(return_value=True)
        
        result = manager.check_resource_limits("test_operation")
        
        assert result is True

    def test_check_resource_limits_train_model_within_limits(self):
        """Test resource limit checking for train_model within limits."""
        manager = ResourceManager(self.mock_config)
        
        # Mock memory check to succeed
        manager.memory_manager.check_memory_limit = Mock(return_value=True)
        
        result = manager.check_resource_limits(
            "train_model",
            sample_count=5000,
            feature_count=500
        )
        
        assert result is True

    def test_check_resource_limits_train_model_sample_limit_exceeded(self):
        """Test resource limit checking for train_model with sample limit exceeded."""
        manager = ResourceManager(self.mock_config)
        
        # Mock memory check to succeed
        manager.memory_manager.check_memory_limit = Mock(return_value=True)
        
        with pytest.raises(ResourceError):
            manager.check_resource_limits(
                "train_model",
                sample_count=15000,  # Exceeds max_samples (10000)
                feature_count=500
            )

    def test_check_resource_limits_train_model_feature_limit_exceeded(self):
        """Test resource limit checking for train_model with feature limit exceeded."""
        manager = ResourceManager(self.mock_config)
        
        # Mock memory check to succeed
        manager.memory_manager.check_memory_limit = Mock(return_value=True)
        
        with pytest.raises(ResourceError):
            manager.check_resource_limits(
                "train_model",
                sample_count=5000,
                feature_count=1500  # Exceeds max_features (1000)
            )

    def test_check_resource_limits_memory_limit_exceeded(self):
        """Test resource limit checking with memory limit exceeded."""
        manager = ResourceManager(self.mock_config)
        
        # Mock memory check to fail
        manager.memory_manager.check_memory_limit = Mock(
            side_effect=MemoryLimitError("Memory limit exceeded", details={})
        )
        
        with pytest.raises(MemoryLimitError):
            manager.check_resource_limits("test_operation")

    def test_check_resource_limits_error_handling(self):
        """Test resource limit checking error handling."""
        manager = ResourceManager(self.mock_config)
        
        # Mock memory check to raise unexpected error
        manager.memory_manager.check_memory_limit = Mock(
            side_effect=Exception("Unexpected error")
        )
        
        with pytest.raises(Exception):
            manager.check_resource_limits("test_operation")


class TestGlobalFunctions:
    """Test global functions."""

    def test_get_resource_manager_singleton(self):
        """Test that get_resource_manager returns singleton instance."""
        # Clear the global instance
        import src.resource_manager
        src.resource_manager._resource_manager = None
        
        manager1 = get_resource_manager()
        manager2 = get_resource_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, ResourceManager)

    def test_check_memory_usage(self):
        """Test check_memory_usage function."""
        with patch('src.resource_manager.get_resource_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.memory_manager.get_memory_usage.return_value = {"rss_mb": 512}
            mock_get_manager.return_value = mock_manager
            
            result = check_memory_usage()
            
            assert result == {"rss_mb": 512}
            mock_manager.memory_manager.get_memory_usage.assert_called_once()

    def test_enforce_limits_decorator(self):
        """Test enforce_limits decorator."""
        with patch('src.resource_manager.get_resource_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.check_resource_limits = Mock(return_value=True)
            
            # Create a proper async context manager mock
            @asynccontextmanager
            async def mock_acquire_resources(operation):
                yield
            
            mock_manager.acquire_resources = mock_acquire_resources
            mock_get_manager.return_value = mock_manager
            
            @enforce_limits("test_operation")
            async def test_function(arg1, arg2):
                return f"result: {arg1}, {arg2}"
            
            # Test the decorated function
            async def run_test():
                result = await test_function("test1", "test2")
                assert result == "result: test1, test2"
                mock_manager.check_resource_limits.assert_called_once_with("test_operation")
            
            # Run the test
            asyncio.run(run_test())

    def test_enforce_limits_decorator_train_model(self):
        """Test enforce_limits decorator with train_model operation."""
        with patch('src.resource_manager.get_resource_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.check_resource_limits = Mock(return_value=True)
            
            @asynccontextmanager
            async def mock_acquire_resources(operation):
                yield
            
            mock_manager.acquire_resources = mock_acquire_resources
            mock_get_manager.return_value = mock_manager
            
            @enforce_limits("train_model")
            async def train_model_function(model_name, data):
                return f"trained: {model_name}"
            
            # Test the decorated function
            async def run_test():
                # Mock data that would be parsed
                test_data = '{"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [7, 8, 9]}'
                
                result = await train_model_function("test_model", test_data)
                assert result == "trained: test_model"
                
                # Should have been called with extracted parameters
                mock_manager.check_resource_limits.assert_called_once()
                call_args = mock_manager.check_resource_limits.call_args
                assert call_args[0][0] == "train_model"
            
            # Run the test
            asyncio.run(run_test())

    def test_enforce_limits_decorator_with_error_handling(self):
        """Test enforce_limits decorator with error handling."""
        with patch('src.resource_manager.get_resource_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.check_resource_limits = Mock(return_value=True)
            
            @asynccontextmanager
            async def mock_acquire_resources(operation):
                yield
            
            mock_manager.acquire_resources = mock_acquire_resources
            mock_get_manager.return_value = mock_manager
            
            @enforce_limits("test_operation")
            async def test_function_with_error():
                raise ValueError("Test error")
            
            # Test the decorated function
            async def run_test():
                with pytest.raises(ValueError):
                    await test_function_with_error()
            
            # Run the test
            asyncio.run(run_test())

    def test_enforce_limits_decorator_json_parsing_error(self):
        """Test enforce_limits decorator with JSON parsing error."""
        with patch('src.resource_manager.get_resource_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.check_resource_limits = Mock(return_value=True)
            
            @asynccontextmanager
            async def mock_acquire_resources(operation):
                yield
            
            mock_manager.acquire_resources = mock_acquire_resources
            mock_get_manager.return_value = mock_manager
            
            @enforce_limits("train_model")
            async def train_model_function(model_name, invalid_data):
                return f"trained: {model_name}"
            
            # Test the decorated function with invalid JSON
            async def run_test():
                result = await train_model_function("test_model", "invalid json")
                assert result == "trained: test_model"
                
                # Should still be called, but without extracted parameters
                mock_manager.check_resource_limits.assert_called_once_with("train_model")
            
            # Run the test
            asyncio.run(run_test()) 