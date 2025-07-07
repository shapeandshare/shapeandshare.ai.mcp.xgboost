#!/usr/bin/env python3

import asyncio
import os
import json
from pathlib import Path
from fastmcp import FastMCP

# Create FastMCP server
mcp = FastMCP("Filesystem")

@mcp.tool()
def read_file(path: str) -> str:
    """Read contents of a file"""
    try:
        # Ensure we're working within the data directory
        data_dir = Path("./data").resolve()
        file_path = Path(path).resolve()
        
        # Security check - ensure file is within data directory
        if not str(file_path).startswith(str(data_dir)):
            file_path = data_dir / Path(path).name
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file"""
    try:
        # Ensure we're working within the data directory
        data_dir = Path("./data").resolve()
        file_path = Path(path).resolve()
        
        # Security check - ensure file is within data directory
        if not str(file_path).startswith(str(data_dir)):
            file_path = data_dir / Path(path).name
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@mcp.tool()
def list_directory(path: str = ".") -> str:
    """List contents of a directory"""
    try:
        # Ensure we're working within the data directory
        data_dir = Path("./data").resolve()
        if path == ".":
            dir_path = data_dir
        else:
            dir_path = Path(path).resolve()
            if not str(dir_path).startswith(str(data_dir)):
                dir_path = data_dir / Path(path).name
        
        if not dir_path.exists():
            return f"Directory does not exist: {dir_path}"
        
        items = []
        for item in sorted(dir_path.iterdir()):
            item_type = "directory" if item.is_dir() else "file"
            size = item.stat().st_size if item.is_file() else "-"
            items.append(f"{item_type}: {item.name} ({size} bytes)")
        
        return f"Contents of {dir_path}:\n" + "\n".join(items)
    except Exception as e:
        return f"Error listing directory: {str(e)}"

@mcp.tool()
def create_directory(path: str) -> str:
    """Create a directory"""
    try:
        # Ensure we're working within the data directory
        data_dir = Path("./data").resolve()
        dir_path = Path(path).resolve()
        
        # Security check - ensure directory is within data directory
        if not str(dir_path).startswith(str(data_dir)):
            dir_path = data_dir / Path(path).name
        
        dir_path.mkdir(parents=True, exist_ok=True)
        return f"Successfully created directory: {dir_path}"
    except Exception as e:
        return f"Error creating directory: {str(e)}"

@mcp.tool()
def delete_file(path: str) -> str:
    """Delete a file"""
    try:
        # Ensure we're working within the data directory
        data_dir = Path("./data").resolve()
        file_path = Path(path).resolve()
        
        # Security check - ensure file is within data directory
        if not str(file_path).startswith(str(data_dir)):
            file_path = data_dir / Path(path).name
        
        if file_path.exists():
            file_path.unlink()
            return f"Successfully deleted file: {file_path}"
        else:
            return f"File does not exist: {file_path}"
    except Exception as e:
        return f"Error deleting file: {str(e)}"

@mcp.tool()
def file_exists(path: str) -> str:
    """Check if a file exists"""
    try:
        # Ensure we're working within the data directory
        data_dir = Path("./data").resolve()
        file_path = Path(path).resolve()
        
        # Security check - ensure file is within data directory
        if not str(file_path).startswith(str(data_dir)):
            file_path = data_dir / Path(path).name
        
        exists = file_path.exists()
        file_type = "directory" if file_path.is_dir() else "file" if file_path.is_file() else "unknown"
        return f"Path {file_path} exists: {exists} (type: {file_type})"
    except Exception as e:
        return f"Error checking file existence: {str(e)}"

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Run the FastMCP server
    mcp.run() 