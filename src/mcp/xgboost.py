"""MCP XGBoost Server Implementation.

This module implements the MCP (Model Context Protocol) server for XGBoost
machine learning operations. It provides HTTP-based endpoints for health
monitoring, sample data generation, and MCP protocol communication.

The server uses FastMCP 2.0 framework to provide a standardized interface
for AI assistants to interact with XGBoost machine learning capabilities.

Functions
---------
health_check
    HTTP endpoint for server health monitoring
sample_data_endpoint
    HTTP endpoint for generating sample data for demonstrations

Module Variables
----------------
mcp : FastMCP
    The main MCP server instance configured for XGBoost operations

Notes
-----
This module serves as the entry point for the MCP server, providing both
standard MCP protocol endpoints and custom HTTP routes for additional
functionality.

The server is designed to be lightweight and efficient, suitable for
deployment in various environments including local development and production setups.

See Also
--------
fastmcp : FastMCP framework for MCP server implementation
src.services.xgboost : XGBoost service implementation
src.utils : Utility functions including prepare_sample_data
"""

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..utils import prepare_sample_data

# Initialize FastMCP 2.0 server with XGBoost capabilities
mcp: FastMCP = FastMCP("MCP XGBoost Server")
"""FastMCP: Main MCP server instance for XGBoost operations.

This server instance handles MCP protocol communication and provides
HTTP endpoints for health monitoring and sample data generation.
"""


# Add a health check endpoint using FastMCP's @custom_route decorator
@mcp.custom_route("/health", methods=["GET"])
async def health_check(_: Request) -> JSONResponse:
    """Health check endpoint for monitoring and load balancers.

    Provides a simple health check endpoint that returns server status
    and basic information about the service. This endpoint is useful
    for monitoring systems, load balancers, and deployment health checks.

    Parameters
    ----------
    _ : Request
        HTTP request object (unused in this endpoint)

    Returns
    -------
    JSONResponse
        JSON response containing:

        - status: Health status ("healthy")
        - service: Service name
        - version: MCP version
        - transport: Transport protocol
        - description: Service description

    Examples
    --------
    Check server health:

        $ curl http://localhost:8000/health
        {
            "status": "healthy",
            "service": "MCP XGBoost Server",
            "version": "2.0",
            "transport": "http",
            "description": "Machine Learning model training and prediction service using XGBoost"
        }

    Notes
    -----
    This endpoint always returns a 200 status code with health information.
    It does not perform deep health checks of dependencies or services.

    See Also
    --------
    fastmcp.FastMCP.custom_route : Custom route decorator
    """
    return JSONResponse(
        {
            "status": "healthy",
            "service": "MCP XGBoost Server",
            "version": "2.0",
            "transport": "http",
            "description": "Machine Learning model training and prediction service using XGBoost",
        }
    )


# Add a sample data endpoint for easy testing
@mcp.custom_route("/sample-data", methods=["GET"])
async def sample_data_endpoint(_: Request) -> JSONResponse:
    """Provide sample data for XGBoost demonstration.

    Generates and returns sample regression data that can be used for
    testing, demonstrations, and examples. The data includes 3 features
    and 1 target variable with known relationships.

    Parameters
    ----------
    _ : Request
        HTTP request object (unused in this endpoint)

    Returns
    -------
    JSONResponse
        JSON response containing:

        - status: Success or error status
        - data: Sample dataset (if successful)
        - description: Description of the data
        - error: Error message (if failed)

    Examples
    --------
    Get sample data:

        $ curl http://localhost:8000/sample-data
        {
            "status": "success",
            "data": {
                "feature1": [1.5, -0.2, ...],
                "feature2": [0.8, 1.1, ...],
                "feature3": [-0.5, 0.3, ...],
                "target": [2.1, 1.4, ...]
            },
            "description": "Sample regression dataset with 3 features and target variable"
        }

    Notes
    -----
    The sample data is generated using a fixed random seed for consistency.
    The target variable has a known relationship to the features, making
    it suitable for testing XGBoost training and prediction capabilities.

    Raises
    ------
    Returns error response if sample data generation fails due to
    ValueError or KeyError exceptions.

    See Also
    --------
    src.utils.prepare_sample_data : Sample data generation function
    """
    try:
        sample_data_dict = prepare_sample_data()
        return JSONResponse(
            {
                "status": "success",
                "data": sample_data_dict,
                "description": "Sample regression dataset with 3 features and target variable",
            }
        )
    except (ValueError, KeyError) as e:
        return JSONResponse({"status": "error", "error": str(e)})
