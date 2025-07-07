"""Services Package for XGBoost Machine Learning Operations.

This package contains service implementations for machine learning operations
using XGBoost. It provides high-level interfaces for model training, prediction,
and analysis, integrated with the MCP (Model Context Protocol) server.

The services package encapsulates all machine learning logic and provides
a clean interface for the MCP server to expose ML capabilities to AI assistants
and other clients.

Modules
-------
xgboost
    Main XGBoost service implementation with comprehensive ML operations
    including model training, prediction, feature importance analysis,
    and model management

Classes
-------
XGBoostService
    Primary service class providing XGBoost machine learning capabilities
    through MCP tool registration and implementation

Notes
-----
This package is designed to be extensible for additional machine learning
services and frameworks. Each service registers its tools with the MCP
server to provide standardized access to ML capabilities.

The services implement comprehensive error handling, data validation,
and user-friendly output formatting for integration with AI assistants.

See Also
--------
src.mcp : MCP server implementation
src.utils : Utility functions for ML operations
xgboost : XGBoost machine learning library
"""
