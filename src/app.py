"""MCP XGBoost Server Application Entry Point.

This module provides the main entry point for the MCP XGBoost server application.
It handles server initialization, configuration, and startup using the FastMCP
framework with HTTP transport.

The application supports environment-based configuration for host and port settings,
and includes comprehensive logging for debugging and monitoring. Health check
endpoints are provided for Kubernetes readiness and liveness probes.

Environment Variables
--------------------
MCP_HOST : str, optional
    Host address for the server (default: "127.0.0.1")
MCP_PORT : str, optional
    Port number for the server (default: "8000")
MCP_STORAGE_BACKEND : str, optional
    Storage backend type ("file" or "redis", default: "file")

Examples
--------
Run the server with default settings:

    $ python -m src.app

Run the server with custom host and port:

    $ MCP_HOST=0.0.0.0 MCP_PORT=9000 python -m src.app

Run the server with Redis storage:

    $ MCP_STORAGE_BACKEND=redis MCP_REDIS_HOST=redis-service python -m src.app

Notes
-----
The server runs with debug logging enabled by default. The MCP endpoint
is available at http://host:port/mcp/ once the server is running.

Health check endpoints:
- /health - Basic health check
- /health/ready - Readiness probe (checks storage connectivity)
- /health/live - Liveness probe (checks application status)

See Also
--------
src.context : Application context and service management
fastmcp : FastMCP framework for MCP server implementation
"""

import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime, UTC

from .context import context

# Health check endpoints
health_app = FastAPI(title="Health Check", description="Application health monitoring")

# Cache for storage connectivity status
_storage_health_cache = {"status": None, "last_check": None, "ttl": 30}

@health_app.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy", "service": "mcp-xgboost", "timestamp": datetime.now(UTC).isoformat()}

@health_app.get("/health/ready")
async def readiness_probe():
    """Readiness probe - checks if the application is ready to serve requests."""
    try:
        from .config import get_config
        
        config = get_config()
        storage_status = await _check_storage_health(config)
        
        return {
            "status": "ready",
            "service": "mcp-xgboost",
            "storage": storage_status,
            "backend": config.storage.storage_backend,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")

@health_app.get("/health/live")
async def liveness_probe():
    """Liveness probe - checks if the application is alive."""
    try:
        # Simple check - if we can respond, we're alive
        import psutil
        import os
        
        # Check basic system resources
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Alert if resources are critically low
        if memory_percent > 95 or cpu_percent > 95:
            raise HTTPException(
                status_code=503, 
                detail=f"High resource usage: CPU {cpu_percent}%, Memory {memory_percent}%"
            )
        
        return {
            "status": "alive",
            "service": "mcp-xgboost",
            "process_id": os.getpid(),
            "memory_percent": memory_percent,
            "cpu_percent": cpu_percent,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=503, detail=f"Service not alive: {str(e)}")

async def _check_storage_health(config) -> str:
    """Check storage backend health with caching."""
    import time
    
    # Check cache first
    current_time = time.time()
    if (_storage_health_cache["last_check"] is not None and 
        current_time - _storage_health_cache["last_check"] < _storage_health_cache["ttl"]):
        return _storage_health_cache["status"]
    
    try:
        if config.storage.storage_backend == "redis":
            # Test Redis connectivity efficiently
            try:
                import redis
                redis_config = {
                    "host": config.redis.host,
                    "port": config.redis.port,
                    "password": config.redis.password,
                    "db": config.redis.database,
                    "socket_timeout": 2,  # Short timeout for health check
                    "socket_connect_timeout": 2,
                    "decode_responses": False,
                }
                
                # Remove None values
                redis_config = {k: v for k, v in redis_config.items() if v is not None}
                
                # Quick connection test
                client = redis.Redis(**redis_config)
                client.ping()
                client.close()
                
                storage_status = "redis_connected"
                
            except Exception as e:
                storage_status = f"redis_error: {str(e)}"
                
        else:
            # Test file storage
            try:
                from pathlib import Path
                
                storage_path = Path(config.storage.model_storage_path)
                # Check if directory exists and is writable
                if not storage_path.exists():
                    storage_path.mkdir(parents=True, exist_ok=True)
                
                # Test write access
                test_file = storage_path / ".health_check"
                test_file.write_text("health_check")
                test_file.unlink()
                
                storage_status = "file_storage_ready"
                
            except Exception as e:
                storage_status = f"file_storage_error: {str(e)}"
        
        # Update cache
        _storage_health_cache["status"] = storage_status
        _storage_health_cache["last_check"] = current_time
        
        return storage_status
        
    except Exception as e:
        return f"storage_check_error: {str(e)}"

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
    logging.info("Health checks available at:")
    logging.info("  - http://%s:%s/health", host, port)
    logging.info("  - http://%s:%s/health/ready", host, port)
    logging.info("  - http://%s:%s/health/live", host, port)

    # Mount health endpoints
    context.mcp.app.mount("/health", health_app)

    # Use the new FastMCP 2.0 API with proper host/port support
    context.mcp.run(transport="http", host=host, port=port)
