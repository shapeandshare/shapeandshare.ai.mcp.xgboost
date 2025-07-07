# Generic Python development workflow with conda
# This file contains reusable targets for Python projects using conda

# Define conda executable and environment
CONDA = conda
CONDA_ENV_PATH ?= ./env
CONDA_ENV_NAME ?= $(shell basename $(CURDIR))
SHELL := /bin/bash

# Ensure conda is initialized and available
.ONESHELL:
export PATH := $(CONDA_PREFIX)/bin:$(PATH)
export CONDA_DEFAULT_ENV := $(CONDA_ENV_NAME)

# Default environment file path (can be overridden)
CONDA_ENV_FILE ?= ./environments/development.yml

# Default source directory (can be overridden)
SRC_DIR ?= src

# Initialize conda in the current shell
init-conda:
	@echo "Initializing conda..."
	@if ! command -v $(CONDA) &> /dev/null; then \
		echo "Error: conda not found. Please install conda first."; \
		exit 1; \
	fi
	@$(CONDA) init bash
	@echo "Conda initialized successfully"

setup: init-conda ## Create the conda environment
	@echo "Creating conda environment..."
	$(CONDA) env create -p $(CONDA_ENV_PATH) -f $(CONDA_ENV_FILE)
	@echo "Environment created successfully"

# Update the conda environment
update-env: ## Update the conda environment
	@echo "Updating conda environment..."
	$(CONDA) env update -p $(CONDA_ENV_PATH) -f $(CONDA_ENV_FILE)
	@echo "Environment updated successfully"

# Use conda environment binaries directly
conda = $(CONDA_ENV_PATH)/bin/

lint: ## Run all linting and style checks
	@echo "Running all linting checks..."
	@echo "Running pylint..."
	$(conda)pylint $(SRC_DIR)/
	@echo "Running flake8..."
	$(conda)flake8 $(SRC_DIR)/
	@echo "Running bandit..."
	$(conda)bandit -r $(SRC_DIR)/
	@echo "Checking code formatting..."
	$(conda)black --check $(SRC_DIR)/
	$(conda)isort --check-only $(SRC_DIR)/
	@echo "All linting checks completed"

lint-fix: ## Run linting with automatic fixes
	@echo "Running linting with automatic fixes..."
	@echo "Formatting code with black and isort..."
	$(conda)black $(SRC_DIR)/
	$(conda)isort $(SRC_DIR)/
	@echo "Auto-formatting with autopep8..."
	$(conda)autopep8 --in-place --recursive $(SRC_DIR)/
	@echo "Automatic fixes completed"

# Environment management

lock: ## Create lock files for all conda environments (multi-platform)
	@echo "Creating lock files for conda environments..."
	@PLATFORMS="linux-64 linux-aarch64"; \
	for platform in $$PLATFORMS; do \
		echo ""; \
		echo "📱 Creating lock files for platform: $$platform"; \
		echo "================================================"; \
		for env_file in environments/*.yml; do \
			if [ -f "$$env_file" ]; then \
				base_name=$$(basename "$$env_file" .yml); \
				if [[ "$$base_name" == *".lock"* ]]; then \
					echo "Skipping lock file: $$env_file"; \
					continue; \
				fi; \
				lock_file="environments/$$base_name.lock-$$platform.yml"; \
				temp_env_path="./temp_env_$${base_name}_$$platform"; \
				echo "Processing $$env_file -> $$lock_file"; \
				echo "Creating temporary environment for $$platform..."; \
				CONDA_SUBDIR=$$platform $(CONDA) env create -p "$$temp_env_path" -f "$$env_file" --quiet; \
				echo "Exporting lock file..."; \
				$(CONDA) env export -p "$$temp_env_path" > "$$lock_file.tmp"; \
				sed "s|name: .*|name: $$base_name-lock-$$platform|" "$$lock_file.tmp" > "$$lock_file"; \
				rm "$$lock_file.tmp"; \
				echo "Cleaning up temporary environment..."; \
				$(CONDA) env remove -p "$$temp_env_path" --yes --quiet; \
				echo "✅ Created $$lock_file"; \
			fi; \
		done; \
	done
	@echo ""
	@echo "🎉 Multi-platform lock files created successfully!"

env-info: ## Show environment information
	@echo "Environment path: $(CONDA_ENV_PATH)"
	@echo "Environment name: $(CONDA_ENV_NAME)"
	@echo "Environment file: $(CONDA_ENV_FILE)"
	@echo "Source directory: $(SRC_DIR)"
	@if [ -d "$(CONDA_ENV_PATH)" ]; then \
		echo "Environment exists: Yes"; \
		$(CONDA) list -p $(CONDA_ENV_PATH); \
	else \
		echo "Environment exists: No"; \
	fi



clean: ## Clean up generated files and caches
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/


clean-env: ## Remove the conda environment
	@echo "Removing conda environment..."
	$(CONDA) env remove -p $(CONDA_ENV_PATH)
	@echo "Environment removed successfully"

# Add all targets to .PHONY
.PHONY: setup update-env lint lint-fix clean clean-env init-conda env-info lock 