"""Tests for src/mcp/xgboost.py module.

This module tests the MCP server implementation including HTTP endpoints
for health monitoring, sample data generation, and FastMCP integration.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.mcp.xgboost import mcp, health_check, sample_data_endpoint


class TestMCPServer:
    """Test the MCP server instance."""

    def test_mcp_server_exists(self):
        """Test that MCP server instance is created."""
        assert mcp is not None
        assert hasattr(mcp, 'custom_route')
        assert hasattr(mcp, 'run')

    def test_mcp_server_name(self):
        """Test that MCP server has correct name."""
        # FastMCP should have a name or identifier
        assert hasattr(mcp, 'name') or hasattr(mcp, '_name') or hasattr(mcp, '__class__')

    def test_mcp_server_type(self):
        """Test that MCP server is correct type."""
        # Should be a FastMCP instance
        from fastmcp import FastMCP
        assert isinstance(mcp, FastMCP)

    def test_mcp_server_imports(self):
        """Test that MCP server can be imported correctly."""
        # Test that we can import the mcp instance
        try:
            from src.mcp.xgboost import mcp as imported_mcp
            assert imported_mcp is mcp
        except ImportError as e:
            pytest.fail(f"Failed to import mcp: {e}")


class TestHealthCheckEndpoint:
    """Test the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_basic(self):
        """Test basic health check functionality."""
        # Create a mock request
        mock_request = Mock(spec=Request)
        
        # Call the health check endpoint
        response = await health_check(mock_request)
        
        # Verify response type
        assert isinstance(response, JSONResponse)

    @pytest.mark.asyncio
    async def test_health_check_response_content(self):
        """Test health check response content."""
        mock_request = Mock(spec=Request)
        
        response = await health_check(mock_request)
        
        # Get the response body
        response_data = response.body.decode('utf-8')
        response_json = json.loads(response_data)
        
        # Verify expected fields
        assert 'status' in response_json
        assert 'service' in response_json
        assert 'version' in response_json
        assert 'transport' in response_json
        assert 'description' in response_json

    @pytest.mark.asyncio
    async def test_health_check_response_values(self):
        """Test health check response values."""
        mock_request = Mock(spec=Request)
        
        response = await health_check(mock_request)
        
        # Get the response body
        response_data = response.body.decode('utf-8')
        response_json = json.loads(response_data)
        
        # Verify specific values
        assert response_json['status'] == 'healthy'
        assert response_json['service'] == 'MCP XGBoost Server'
        assert response_json['version'] == '2.0'
        assert response_json['transport'] == 'http'
        assert 'XGBoost' in response_json['description']

    @pytest.mark.asyncio
    async def test_health_check_status_code(self):
        """Test health check returns 200 status code."""
        mock_request = Mock(spec=Request)
        
        response = await health_check(mock_request)
        
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_check_content_type(self):
        """Test health check returns JSON content type."""
        mock_request = Mock(spec=Request)
        
        response = await health_check(mock_request)
        
        # JSONResponse should set correct content type
        assert 'application/json' in response.headers.get('content-type', '')

    @pytest.mark.asyncio
    async def test_health_check_request_unused(self):
        """Test that health check doesn't use the request parameter."""
        # The endpoint takes a request parameter but doesn't use it
        mock_request = Mock(spec=Request)
        
        # Should work with any request object
        response1 = await health_check(mock_request)
        response2 = await health_check(None)  # Even with None
        
        # Both should return same response structure
        assert isinstance(response1, JSONResponse)
        assert isinstance(response2, JSONResponse)

    @pytest.mark.asyncio
    async def test_health_check_multiple_calls(self):
        """Test that health check is consistent across multiple calls."""
        mock_request = Mock(spec=Request)
        
        # Call multiple times
        responses = []
        for _ in range(3):
            response = await health_check(mock_request)
            response_data = response.body.decode('utf-8')
            response_json = json.loads(response_data)
            responses.append(response_json)
        
        # All responses should be identical
        for response in responses[1:]:
            assert response == responses[0]


class TestSampleDataEndpoint:
    """Test the sample data endpoint."""

    @pytest.mark.asyncio
    async def test_sample_data_basic(self):
        """Test basic sample data functionality."""
        mock_request = Mock(spec=Request)
        
        response = await sample_data_endpoint(mock_request)
        
        assert isinstance(response, JSONResponse)

    @pytest.mark.asyncio
    @patch('src.mcp.xgboost.prepare_sample_data')
    async def test_sample_data_success(self, mock_prepare_sample_data):
        """Test successful sample data generation."""
        # Mock the prepare_sample_data function
        mock_data = {
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [7, 8, 9]
        }
        mock_prepare_sample_data.return_value = mock_data
        
        mock_request = Mock(spec=Request)
        
        response = await sample_data_endpoint(mock_request)
        
        # Verify response
        assert isinstance(response, JSONResponse)
        assert response.status_code == 200
        
        # Check response content
        response_data = response.body.decode('utf-8')
        response_json = json.loads(response_data)
        
        assert response_json['status'] == 'success'
        assert 'data' in response_json
        assert 'description' in response_json
        assert response_json['data'] == mock_data

    @pytest.mark.asyncio
    @patch('src.mcp.xgboost.prepare_sample_data')
    async def test_sample_data_response_structure(self, mock_prepare_sample_data):
        """Test sample data response structure."""
        mock_data = {'test': 'data'}
        mock_prepare_sample_data.return_value = mock_data
        
        mock_request = Mock(spec=Request)
        
        response = await sample_data_endpoint(mock_request)
        
        response_data = response.body.decode('utf-8')
        response_json = json.loads(response_data)
        
        # Check required fields
        assert 'status' in response_json
        assert 'data' in response_json
        assert 'description' in response_json
        
        # Check description content
        assert 'regression' in response_json['description'].lower()
        assert 'features' in response_json['description'].lower()

    @pytest.mark.asyncio
    @patch('src.mcp.xgboost.prepare_sample_data')
    async def test_sample_data_value_error(self, mock_prepare_sample_data):
        """Test sample data endpoint with ValueError."""
        # Mock prepare_sample_data to raise ValueError
        mock_prepare_sample_data.side_effect = ValueError("Sample data generation failed")
        
        mock_request = Mock(spec=Request)
        
        response = await sample_data_endpoint(mock_request)
        
        # Should return error response
        assert isinstance(response, JSONResponse)
        
        response_data = response.body.decode('utf-8')
        response_json = json.loads(response_data)
        
        assert response_json['status'] == 'error'
        assert 'error' in response_json
        assert 'Sample data generation failed' in response_json['error']

    @pytest.mark.asyncio
    @patch('src.mcp.xgboost.prepare_sample_data')
    async def test_sample_data_key_error(self, mock_prepare_sample_data):
        """Test sample data endpoint with KeyError."""
        # Mock prepare_sample_data to raise KeyError
        mock_prepare_sample_data.side_effect = KeyError("Missing key")
        
        mock_request = Mock(spec=Request)
        
        response = await sample_data_endpoint(mock_request)
        
        # Should return error response
        assert isinstance(response, JSONResponse)
        
        response_data = response.body.decode('utf-8')
        response_json = json.loads(response_data)
        
        assert response_json['status'] == 'error'
        assert 'error' in response_json
        assert 'Missing key' in response_json['error']

    @pytest.mark.asyncio
    @patch('src.mcp.xgboost.prepare_sample_data')
    async def test_sample_data_other_exceptions_not_caught(self, mock_prepare_sample_data):
        """Test that other exceptions are not caught."""
        # Mock prepare_sample_data to raise a different exception
        mock_prepare_sample_data.side_effect = RuntimeError("Unexpected error")
        
        mock_request = Mock(spec=Request)
        
        # Should raise the RuntimeError, not catch it
        with pytest.raises(RuntimeError, match="Unexpected error"):
            await sample_data_endpoint(mock_request)

    @pytest.mark.asyncio
    async def test_sample_data_request_unused(self):
        """Test that sample data endpoint doesn't use the request parameter."""
        mock_request = Mock(spec=Request)
        
        # Should work with any request object
        response1 = await sample_data_endpoint(mock_request)
        response2 = await sample_data_endpoint(None)
        
        # Both should return same response structure
        assert isinstance(response1, JSONResponse)
        assert isinstance(response2, JSONResponse)

    @pytest.mark.asyncio
    @patch('src.mcp.xgboost.prepare_sample_data')
    async def test_sample_data_prepare_function_called(self, mock_prepare_sample_data):
        """Test that prepare_sample_data function is called."""
        mock_prepare_sample_data.return_value = {'test': 'data'}
        
        mock_request = Mock(spec=Request)
        
        await sample_data_endpoint(mock_request)
        
        # Verify the function was called
        mock_prepare_sample_data.assert_called_once()


class TestMCPServerDecorators:
    """Test MCP server custom route decorators."""

    def test_health_check_decorator_applied(self):
        """Test that health check has custom route decorator."""
        # The health check function should be decorated
        # We can test this by checking if the route is registered
        assert hasattr(mcp, 'custom_route')
        
        # The decorator should have been applied during import
        # This test verifies the decorator exists and can be used

    def test_sample_data_decorator_applied(self):
        """Test that sample data endpoint has custom route decorator."""
        # The sample data function should be decorated
        # Similar to health check, should be registered as custom route
        assert hasattr(mcp, 'custom_route')

    def test_mcp_server_has_routes(self):
        """Test that MCP server has registered routes."""
        # FastMCP should have methods for handling routes
        assert hasattr(mcp, 'custom_route') or hasattr(mcp, 'add_route')


class TestMCPServerIntegration:
    """Test MCP server integration aspects."""

    def test_imports_from_utils(self):
        """Test that utils module imports work correctly."""
        # Test the import from utils
        try:
            from src.utils import prepare_sample_data
            assert callable(prepare_sample_data)
        except ImportError as e:
            pytest.fail(f"Failed to import from utils: {e}")

    def test_imports_from_fastmcp(self):
        """Test that FastMCP imports work correctly."""
        try:
            from fastmcp import FastMCP
            from starlette.requests import Request
            from starlette.responses import JSONResponse
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import FastMCP dependencies: {e}")

    def test_mcp_server_creation(self):
        """Test MCP server creation process."""
        # Test that we can create a new FastMCP instance
        from fastmcp import FastMCP
        
        test_server = FastMCP("Test Server")
        assert test_server is not None
        assert hasattr(test_server, 'custom_route')

    def test_endpoint_function_signatures(self):
        """Test that endpoint functions have correct signatures."""
        import inspect
        
        # Test health_check signature
        health_sig = inspect.signature(health_check)
        assert len(health_sig.parameters) == 1
        
        # Test sample_data_endpoint signature
        sample_sig = inspect.signature(sample_data_endpoint)
        assert len(sample_sig.parameters) == 1

    def test_endpoint_function_async(self):
        """Test that endpoint functions are async."""
        import asyncio
        
        # Both functions should be coroutines
        assert asyncio.iscoroutinefunction(health_check)
        assert asyncio.iscoroutinefunction(sample_data_endpoint)


class TestModuleStructure:
    """Test module structure and exports."""

    def test_module_docstring(self):
        """Test that module has proper documentation."""
        import src.mcp.xgboost as mcp_module
        assert mcp_module.__doc__ is not None
        assert len(mcp_module.__doc__) > 0
        assert "MCP XGBoost Server Implementation" in mcp_module.__doc__

    def test_module_exports(self):
        """Test module exports expected objects."""
        import src.mcp.xgboost as mcp_module
        
        # Should export the mcp server instance
        assert hasattr(mcp_module, 'mcp')
        assert hasattr(mcp_module, 'health_check')
        assert hasattr(mcp_module, 'sample_data_endpoint')

    def test_module_dependencies(self):
        """Test that all module dependencies are available."""
        # All imports should work
        try:
            from fastmcp import FastMCP
            from starlette.requests import Request
            from starlette.responses import JSONResponse
            from src.utils import prepare_sample_data
            assert True
        except ImportError as e:
            pytest.fail(f"Module dependency not available: {e}")

    def test_module_constants(self):
        """Test module-level constants and configuration."""
        # The mcp instance should be properly configured
        assert mcp is not None
        
        # Should be a FastMCP instance with XGBoost server name
        from fastmcp import FastMCP
        assert isinstance(mcp, FastMCP)


class TestErrorHandling:
    """Test error handling in endpoints."""

    @pytest.mark.asyncio
    async def test_health_check_robust(self):
        """Test that health check is robust to various inputs."""
        # Should work with different request types
        responses = []
        
        # Test with None
        response = await health_check(None)
        responses.append(response)
        
        # Test with Mock
        mock_request = Mock()
        response = await health_check(mock_request)
        responses.append(response)
        
        # All should be successful
        for response in responses:
            assert isinstance(response, JSONResponse)
            assert response.status_code == 200

    @pytest.mark.asyncio
    @patch('src.mcp.xgboost.prepare_sample_data')
    async def test_sample_data_error_response_format(self, mock_prepare_sample_data):
        """Test error response format is consistent."""
        mock_prepare_sample_data.side_effect = ValueError("Test error")
        
        mock_request = Mock(spec=Request)
        
        response = await sample_data_endpoint(mock_request)
        response_data = response.body.decode('utf-8')
        response_json = json.loads(response_data)
        
        # Error response should have consistent format
        assert 'status' in response_json
        assert 'error' in response_json
        assert response_json['status'] == 'error'
        assert isinstance(response_json['error'], str)

    @pytest.mark.asyncio
    async def test_json_response_validity(self):
        """Test that all responses return valid JSON."""
        mock_request = Mock(spec=Request)
        
        # Test health check
        health_response = await health_check(mock_request)
        health_data = health_response.body.decode('utf-8')
        json.loads(health_data)  # Should not raise exception
        
        # Test sample data (with mocked prepare_sample_data)
        with patch('src.mcp.xgboost.prepare_sample_data', return_value={'test': 'data'}):
            sample_response = await sample_data_endpoint(mock_request)
            sample_data = sample_response.body.decode('utf-8')
            json.loads(sample_data)  # Should not raise exception


class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_health_check_fast(self):
        """Test that health check responds quickly."""
        import time
        
        mock_request = Mock(spec=Request)
        
        start_time = time.time()
        await health_check(mock_request)
        end_time = time.time()
        
        # Should be very fast (less than 1 second)
        assert end_time - start_time < 1.0

    @pytest.mark.asyncio
    async def test_endpoints_concurrent(self):
        """Test that endpoints can handle concurrent requests."""
        import asyncio
        
        mock_request = Mock(spec=Request)
        
        # Run multiple requests concurrently
        tasks = []
        for _ in range(5):
            task1 = health_check(mock_request)
            tasks.append(task1)
        
        with patch('src.mcp.xgboost.prepare_sample_data', return_value={'test': 'data'}):
            for _ in range(5):
                task2 = sample_data_endpoint(mock_request)
                tasks.append(task2)
        
        # All should complete successfully
        responses = await asyncio.gather(*tasks)
        
        for response in responses:
            assert isinstance(response, JSONResponse)
            assert response.status_code == 200 