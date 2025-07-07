#!/bin/bash

# Multi-platform Docker build script for MCP XGBoost
# This script uses the integrated make commands for building multi-platform images

set -e

echo "🚀 Multi-platform Docker Build for MCP XGBoost"
echo "=============================================="
echo "This script uses the integrated 'make app-build' command"
echo ""

# Use the integrated make command
make app-build

echo ""
echo "🎉 Multi-platform build completed successfully!"
echo ""
echo "📋 Available commands:"
echo "  make app-build        - Build multi-platform image (linux/amd64, linux/arm64)"
echo "  make app-build-local  - Build for current platform only (faster for development)"
echo "  make app-install      - Install to k3d cluster"
echo "  make app-deploy       - Build + install in one command"
echo ""
echo "📋 To test the image:"
echo "  docker run --rm mcp-xgboost:latest"
echo ""
echo "🧹 To clean up buildx:"
echo "  docker buildx rm multiplatform-builder" 