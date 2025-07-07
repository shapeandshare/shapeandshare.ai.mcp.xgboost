# Project-specific Makefile for mcp-xgboost
# This file imports shared functionality and defines project-specific configurations

# ====================================================================
# 🚀 DEVELOPMENT ENVIRONMENT - SIMPLIFIED INTERFACE
# ====================================================================
# Use these commands for your development environment:
#
#   make dev-setup   - Create complete dev environment (k3d + Dashboard)
#   make dev-reset    - Reset dev environment (clean rebuild)
#   make dev-nuke  - Destroy dev environment completely
#   make dev-access   - Access dashboard and services
#
# After 'make dev-setup', use 'make dev-access' for URLs and tokens
# ====================================================================

# Include shared functionality
include shared/helper.mk
include shared/python.mk
include shared/dev-env.mk

# Legacy includes (available but use simplified dev-env.mk instead)
include shared/k3d.mk
include shared/k8s.mk

# Project-specific configurations (override defaults if needed)
# SRC_DIR = src  # Already defaults to src in shared/python.mk
# CONDA_ENV_FILE = ./environments/development.yml  # Already defaults to this path

# ====================================================================
# 🔨 BUILD COMMANDS
# ====================================================================

build: ## Smart Docker build (local by default, multi-platform if MULTIPLATFORM=true)
	@# Import shared color definitions from dev-env.mk
	@if [ "$(MULTIPLATFORM)" = "true" ] || [ "$(CI)" = "true" ]; then \
		echo "🌐 Building multi-platform image for CI/CD..."; \
		$(MAKE) app-build; \
	else \
		echo "🔨 Building local platform image for development..."; \
		$(MAKE) app-build-local; \
	fi
	@echo ""
	@echo "💡 Build options:"
	@echo "  make build                    - Local platform (default)"
	@echo "  make build MULTIPLATFORM=true - Multi-platform (CI/CD)"
	@echo "  make app-build-local          - Force local build"
	@echo "  make app-build                - Force multi-platform build"

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

# Development environment is managed by shared/dev-env.mk (simplified interface)
# Use: dev-setup, dev-reset, dev-nuke, dev-access, dev-status
# 
# Legacy Kubernetes commands are still available in shared/k8s.mk, shared/k3d.mk
# But it's recommended to use the simplified dev-* commands above

# Add project-specific targets to .PHONY if any are defined
.PHONY: build run unit-test

# Development environment targets from shared/dev-env.mk
.PHONY: dev-setup dev-reset dev-nuke dev-access dev-cleanup app-build app-build-local app-install app-deploy app-redeploy app-uninstall
