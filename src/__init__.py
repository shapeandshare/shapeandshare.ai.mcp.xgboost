"""MCP XGBoost Server Package.

This package provides a Model Context Protocol (MCP) server implementation for
XGBoost machine learning functionality. It enables training, prediction, and
analysis of XGBoost models through a standardized MCP interface.

The package consists of:
- MCP server setup and configuration
- XGBoost service implementation with comprehensive ML operations
- Utility functions for data validation and model management
- Context management for service dependencies

Examples
--------
To run the MCP server:

    $ python -m src.app

To use the service programmatically:

    >>> from src.context import context
    >>> service = context.service

Notes
-----
This package requires XGBoost, pandas, numpy, and FastMCP dependencies.
It's designed to work with the MCP 2.0 specification for AI assistant
integration.

See Also
--------
fastmcp : FastMCP framework for MCP server implementation
xgboost : XGBoost machine learning library
"""
