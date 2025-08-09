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

# ====================================================================
# 🔧 CLAUDE DESKTOP INTEGRATION
# ====================================================================

install-claude: ## Install MCP server configuration to Claude Desktop
	@echo "Installing MCP XGBoost configuration to Claude Desktop..."
	@CLAUDE_CONFIG_DIR="$$HOME/Library/Application Support/Claude"; \
	CLAUDE_CONFIG_FILE="$$CLAUDE_CONFIG_DIR/claude_desktop_config.json"; \
	PROJECT_CONFIG="$(CURDIR)/claude-desktop-config.json"; \
	if [ ! -f "$$PROJECT_CONFIG" ]; then \
		echo "❌ Error: claude-desktop-config.json not found in project root"; \
		exit 1; \
	fi; \
	mkdir -p "$$CLAUDE_CONFIG_DIR"; \
	if [ -f "$$CLAUDE_CONFIG_FILE" ]; then \
		echo "📋 Backing up existing Claude Desktop config..."; \
		cp "$$CLAUDE_CONFIG_FILE" "$$CLAUDE_CONFIG_FILE.backup.$$(date +%Y%m%d_%H%M%S)"; \
		echo "✅ Backup created: $$CLAUDE_CONFIG_FILE.backup.$$(date +%Y%m%d_%H%M%S)"; \
		echo "🔄 Merging with existing configuration..."; \
		python3 -c "import json, sys; \
		existing = json.load(open('$$CLAUDE_CONFIG_FILE')); \
		new = json.load(open('$$PROJECT_CONFIG')); \
		existing.setdefault('mcpServers', {}).update(new.get('mcpServers', {})); \
		json.dump(existing, open('$$CLAUDE_CONFIG_FILE', 'w'), indent=2)"; \
	else \
		echo "📝 Creating new Claude Desktop config..."; \
		cp "$$PROJECT_CONFIG" "$$CLAUDE_CONFIG_FILE"; \
	fi; \
	echo "✅ MCP XGBoost server installed to Claude Desktop!"; \
	echo "📍 Config location: $$CLAUDE_CONFIG_FILE"; \
	echo "🔄 Please restart Claude Desktop to load the new configuration."

uninstall-claude: ## Remove MCP server configuration from Claude Desktop
	@echo "Removing MCP XGBoost configuration from Claude Desktop..."
	@CLAUDE_CONFIG_DIR="$$HOME/Library/Application Support/Claude"; \
	CLAUDE_CONFIG_FILE="$$CLAUDE_CONFIG_DIR/claude_desktop_config.json"; \
	if [ -f "$$CLAUDE_CONFIG_FILE" ]; then \
		echo "📋 Backing up current Claude Desktop config..."; \
		cp "$$CLAUDE_CONFIG_FILE" "$$CLAUDE_CONFIG_FILE.backup.$$(date +%Y%m%d_%H%M%S)"; \
	fi; \
	python3 scripts/claude_uninstall.py; \
	echo "🔄 Please restart Claude Desktop to apply the changes."

claude-status: ## Check Claude Desktop configuration status
	@echo "Checking Claude Desktop configuration status..."
	@python3 scripts/claude_status.py

# Add project-specific targets to .PHONY if any are defined
.PHONY: run unit-test install-claude uninstall-claude claude-status
