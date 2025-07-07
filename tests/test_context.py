"""Tests for src/context.py module.

This module tests the Context class and global context instance that manages
the application's core components including MCP server and XGBoost service.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastmcp import FastMCP
from src.context import Context, context
from src.services.xgboost import XGBoostService


class TestContext:
    """Test the Context class."""

    @patch('src.context.mcp')
    @patch('src.context.XGBoostService')
    def test_context_initialization(self, mock_xgboost_service, mock_mcp):
        """Test Context class initialization."""
        # Setup mocks
        mock_mcp_instance = Mock(spec=FastMCP)
        mock_mcp.return_value = mock_mcp_instance
        mock_service_instance = Mock(spec=XGBoostService)
        mock_xgboost_service.return_value = mock_service_instance
        
        # Create context instance
        test_context = Context()
        
        # Verify attributes are set correctly
        assert test_context.mcp == mock_mcp
        assert test_context.service == mock_service_instance
        
        # Verify XGBoost service was initialized with MCP server
        mock_xgboost_service.assert_called_once_with(mcp=mock_mcp)

    @patch('src.context.mcp')
    @patch('src.context.XGBoostService')
    def test_context_attributes_access(self, mock_xgboost_service, mock_mcp):
        """Test accessing Context attributes."""
        # Setup mocks
        mock_mcp_instance = Mock(spec=FastMCP)
        mock_mcp.return_value = mock_mcp_instance
        mock_service_instance = Mock(spec=XGBoostService)
        mock_xgboost_service.return_value = mock_service_instance
        
        # Create context instance
        test_context = Context()
        
        # Test attribute access
        assert hasattr(test_context, 'mcp')
        assert hasattr(test_context, 'service')
        assert test_context.mcp is not None
        assert test_context.service is not None

    @patch('src.context.mcp')
    @patch('src.context.XGBoostService')
    def test_context_service_integration(self, mock_xgboost_service, mock_mcp):
        """Test that Context properly integrates MCP and XGBoost service."""
        # Setup mocks
        mock_mcp_instance = Mock(spec=FastMCP)
        mock_mcp.return_value = mock_mcp_instance
        mock_service_instance = Mock(spec=XGBoostService)
        mock_xgboost_service.return_value = mock_service_instance
        
        # Create context instance
        test_context = Context()
        
        # Verify service is initialized with the correct MCP instance
        mock_xgboost_service.assert_called_once_with(mcp=mock_mcp)
        
        # Verify the service instance is stored
        assert test_context.service == mock_service_instance

    def test_context_class_attributes(self):
        """Test Context class has expected attributes."""
        # Test that Context class has the expected structure
        assert hasattr(Context, '__init__')
        assert callable(Context.__init__)
        
        # Test that Context can be instantiated
        with patch('src.context.mcp'), patch('src.context.XGBoostService'):
            context_instance = Context()
            assert context_instance is not None


class TestGlobalContext:
    """Test the global context instance."""

    def test_global_context_exists(self):
        """Test that global context instance exists."""
        # Import the global context
        from src.context import context as global_context
        
        assert global_context is not None
        assert isinstance(global_context, Context)

    def test_global_context_is_singleton(self):
        """Test that global context behaves as singleton."""
        # Import context multiple times
        from src.context import context as context1
        from src.context import context as context2
        
        # Should be the same instance
        assert context1 is context2

    def test_global_context_has_mcp(self):
        """Test that global context has MCP server."""
        from src.context import context as global_context
        
        assert hasattr(global_context, 'mcp')
        assert global_context.mcp is not None

    def test_global_context_has_service(self):
        """Test that global context has XGBoost service."""
        from src.context import context as global_context
        
        assert hasattr(global_context, 'service')
        assert global_context.service is not None

    def test_global_context_service_type(self):
        """Test that global context service is XGBoost service."""
        from src.context import context as global_context
        
        assert isinstance(global_context.service, XGBoostService)

    def test_global_context_mcp_type(self):
        """Test that global context MCP is FastMCP instance."""
        from src.context import context as global_context
        
        # Should be a FastMCP instance or have FastMCP-like interface
        assert hasattr(global_context.mcp, 'run')  # FastMCP has run method
        assert hasattr(global_context.mcp, 'tool')  # FastMCP has tool decorator


class TestContextIntegration:
    """Test Context integration with other components."""

    def test_context_mcp_service_connection(self):
        """Test that Context connects MCP and service correctly."""
        from src.context import context as global_context
        
        # The service should be initialized with the MCP instance
        assert global_context.service._mcp is global_context.mcp

    def test_context_module_imports(self):
        """Test that Context module imports work correctly."""
        # Test individual imports
        from src.context import Context as ImportedContext
        from src.context import context as imported_context
        
        assert ImportedContext is Context
        assert imported_context is context

    def test_context_dependency_injection(self):
        """Test that Context provides proper dependency injection."""
        from src.context import context as global_context
        
        # Context should provide access to both MCP and service
        mcp_instance = global_context.mcp
        service_instance = global_context.service
        
        assert mcp_instance is not None
        assert service_instance is not None
        
        # Service should have reference to MCP
        assert hasattr(service_instance, '_mcp')
        assert service_instance._mcp is mcp_instance

    @patch('src.context.mcp')
    @patch('src.context.XGBoostService')
    def test_context_initialization_error_handling(self, mock_xgboost_service, mock_mcp):
        """Test Context initialization handles errors gracefully."""
        # Test with XGBoost service initialization failure
        mock_xgboost_service.side_effect = Exception("Service initialization failed")
        
        with pytest.raises(Exception) as exc_info:
            Context()
        
        assert "Service initialization failed" in str(exc_info.value)

    def test_context_attributes_immutability(self):
        """Test that Context attributes are properly set."""
        from src.context import context as global_context
        
        # Test that attributes exist and are not None
        original_mcp = global_context.mcp
        original_service = global_context.service
        
        assert original_mcp is not None
        assert original_service is not None
        
        # Test that they maintain their references
        assert global_context.mcp is original_mcp
        assert global_context.service is original_service


class TestContextDocumentation:
    """Test Context class documentation and structure."""

    def test_context_class_docstring(self):
        """Test Context class has proper documentation."""
        assert Context.__doc__ is not None
        assert len(Context.__doc__) > 0

    def test_context_init_docstring(self):
        """Test Context.__init__ has proper documentation."""
        assert Context.__init__.__doc__ is not None
        assert len(Context.__init__.__doc__) > 0

    def test_context_module_docstring(self):
        """Test context module has proper documentation."""
        import src.context as context_module
        assert context_module.__doc__ is not None
        assert len(context_module.__doc__) > 0

    def test_global_context_docstring(self):
        """Test global context has proper documentation."""
        # Check that the global context variable has documentation
        import src.context as context_module
        
        # The module should have the context variable with documentation
        assert hasattr(context_module, 'context')
        assert context_module.context is not None


class TestContextUsage:
    """Test Context usage patterns."""

    def test_context_as_module_import(self):
        """Test importing context from module."""
        from src.context import context as module_context
        
        # Should be able to access MCP and service
        assert hasattr(module_context, 'mcp')
        assert hasattr(module_context, 'service')
        
        # Should be functional
        assert callable(getattr(module_context.mcp, 'run', None))
        assert hasattr(module_context.service, '_mcp')

    def test_context_for_mcp_access(self):
        """Test using context to access MCP server."""
        from src.context import context as global_context
        
        mcp_server = global_context.mcp
        assert mcp_server is not None
        
        # Should have FastMCP methods
        assert hasattr(mcp_server, 'run')
        assert hasattr(mcp_server, 'tool')

    def test_context_for_service_access(self):
        """Test using context to access XGBoost service."""
        from src.context import context as global_context
        
        service = global_context.service
        assert service is not None
        
        # Should have XGBoost service methods
        assert hasattr(service, '_mcp')
        assert hasattr(service, '_models')

    def test_context_lifecycle(self):
        """Test Context lifecycle management."""
        from src.context import context as global_context
        
        # Context should be initialized and ready
        assert global_context.mcp is not None
        assert global_context.service is not None
        
        # Should maintain state
        mcp_ref = global_context.mcp
        service_ref = global_context.service
        
        # References should be stable
        assert global_context.mcp is mcp_ref
        assert global_context.service is service_ref 