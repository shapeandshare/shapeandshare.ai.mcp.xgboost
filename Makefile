# Project-specific Makefile for mcp-xgboost
# This file imports shared functionality and defines project-specific configurations

# ====================================================================
# 🚀 DEVELOPMENT ENVIRONMENT
# ====================================================================
# Use these commands for your development environment:
#
#   make lock         - Create conda lock files
#   make install      - Install project dependencies

#   make run          - Run the application locally
#   make unit-test    - Run unit tests with coverage
#
# ====================================================================

# Include shared functionality
include shared/helper.mk
include shared/python.mk

# Project-specific configurations (override defaults if needed)
# SRC_DIR = src  # Already defaults to src in shared/python.mk
# CONDA_ENV_FILE = ./environments/development.yml  # Already defaults to this path

# ====================================================================
# 🔨 APPLICATION COMMANDS
# ====================================================================

run: ## Run the MCP XGBoost application
	@echo "Running mcp-xgboost application..."
	@echo "Press Ctrl+C to stop the server"
	$(conda)python -m src.app

unit-test: ## Run tests with comprehensive coverage reporting
	@echo "================================================"
	@echo "Running tests with coverage..."
	@echo "================================================"
	$(conda)pytest tests/ -v --tb=short --color=yes --cov=$(SRC_DIR)/ --cov-report=term-missing --cov-report=html --cov-report=xml --cov-branch
	@echo ""
	@echo "================================================"
	@echo "📊 Coverage Report Summary:"
	@echo "- HTML Report: htmlcov/index.html"
	@echo "- XML Report: coverage.xml"
	@echo "- Branch coverage included"
	@echo "- View in browser: open htmlcov/index.html"
	@echo "================================================"
	@echo "Tests with coverage completed successfully!"
	@echo "================================================"

# Add project-specific targets to .PHONY if any are defined
.PHONY: run unit-test
