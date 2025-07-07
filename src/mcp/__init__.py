"""MCP (Model Context Protocol) Server Implementation Package.

This package contains the MCP server implementation for the XGBoost service,
providing HTTP-based protocol communication for machine learning operations.
It implements the MCP 2.0 specification using the FastMCP framework.

The package includes:
- MCP server initialization and configuration
- HTTP route handlers for health checks and sample data
- Integration with XGBoost service tools

Modules
-------
xgboost
    Main MCP server implementation with custom routes and tool registration

Notes
-----
This package implements the MCP 2.0 specification for AI assistant integration,
providing a standardized interface for machine learning operations through
the XGBoost service.

The server runs on HTTP transport and provides endpoints for:
- Health monitoring (/health)
- Sample data generation (/sample-data)
- MCP protocol communication (/mcp/)

See Also
--------
fastmcp : FastMCP framework for MCP server implementation
src.services.xgboost : XGBoost service implementation
"""
