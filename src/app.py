"""MCP XGBoost Server Application Entry Point.

This module provides the main entry point for the MCP XGBoost server application.
It handles server initialization, configuration, and startup using the FastMCP
framework with HTTP transport.

The application supports environment-based configuration for host and port settings,
and includes comprehensive logging for debugging and monitoring.

Environment Variables
--------------------
MCP_HOST : str, optional
    Host address for the server (default: "127.0.0.1")
MCP_PORT : str, optional
    Port number for the server (default: "8000")

Examples
--------
Run the server with default settings:

    $ python -m src.app

Run the server with custom host and port:

    $ MCP_HOST=0.0.0.0 MCP_PORT=9000 python -m src.app

Notes
-----
The server runs with debug logging enabled by default. The MCP endpoint
is available at http://host:port/mcp/ once the server is running.

See Also
--------
src.context : Application context and service management
fastmcp : FastMCP framework for MCP server implementation
"""

import logging
import os

from .context import context

if __name__ == "__main__":
    # Configure logging to show debug output
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get host and port from environment variables or use defaults
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8000"))

    # Initialize and run the server with FastMCP 2.0
    logging.info("Starting MCP XGBoost Server (FastMCP 2.0)...")
    logging.info("Server will be available at http://%s:%s/mcp/", host, port)

    # Use the new FastMCP 2.0 API with proper host/port support
    context.mcp.run(transport="http", host=host, port=port)
