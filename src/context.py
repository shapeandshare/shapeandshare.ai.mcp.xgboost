"""Application Context Management.

This module provides centralized context management for the MCP XGBoost server
application. It initializes and manages the MCP server instance and XGBoost
service, providing a single access point for all application components.

The context pattern ensures proper dependency injection and lifecycle management
for the application's core services.

Classes
-------
Context
    Central application context manager that initializes and provides access
    to the MCP server and XGBoost service instances.

Module Variables
----------------
context : Context
    Global context instance providing access to application services.

Examples
--------
Access the MCP server:

    >>> from src.context import context
    >>> mcp_server = context.mcp

Access the XGBoost service:

    >>> from src.context import context
    >>> xgboost_service = context.service

Notes
-----
The context is initialized as a module-level singleton to ensure consistent
access to services throughout the application lifecycle.

See Also
--------
src.mcp.xgboost : MCP server implementation
src.services.xgboost : XGBoost service implementation
"""

from .mcp.xgboost import mcp
from .services.xgboost import XGBoostService


class Context:
    """Central application context manager.

    This class initializes and manages the core application components,
    providing a centralized access point for the MCP server and XGBoost
    service instances.

    Attributes
    ----------
    mcp : FastMCP
        The MCP server instance for handling protocol communications
    service : XGBoostService
        The XGBoost service instance for machine learning operations

    Examples
    --------
    Create a new context instance:

        >>> context = Context()
        >>> mcp_server = context.mcp
        >>> ml_service = context.service

    Notes
    -----
    The context automatically initializes the XGBoost service with the
    MCP server instance, ensuring proper integration between components.
    """

    def __init__(self) -> None:
        """Initialize the application context.

        Sets up the MCP server and XGBoost service instances, establishing
        the connections between components for proper operation.
        """
        self.mcp = mcp
        self.service = XGBoostService(mcp=self.mcp)


# Global context instance
context = Context()
"""Context: Global application context instance.

This singleton instance provides access to the MCP server and XGBoost service
throughout the application. It should be imported and used wherever these
services are needed.
"""
