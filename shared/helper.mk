# Helper makefile for automatic help generation
# This file should be included in your main Makefile

# ANSI color codes - use environment variable to disable colors if needed
# Set NO_COLOR=1 to disable colors entirely
ifeq ($(NO_COLOR),1)
    BOLD :=
    RESET :=
    BLUE :=
    GREEN :=
else
    BOLD := $(shell printf '\033[1m')
    RESET := $(shell printf '\033[0m')
    BLUE := $(shell printf '\033[0;34m')
    GREEN := $(shell printf '\033[0;32m')
endif

# Default target shows help
.DEFAULT_GOAL := help

# Help target that extracts ## comments from all targets
help: ## Show this help message
	@printf "$(BOLD)$(BLUE)Available commands:$(RESET)\n"
	@printf "\n"
	@cat $(MAKEFILE_LIST) | grep -E '^[a-zA-Z0-9_-]+:.*## .*' | \
		sed 's/\(.*\):\(.*\)## \(.*\)/\1:\3/' | \
		awk -F ':' '{printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}' | \
		sort -u
	@printf "\n"
	@printf "$(BOLD)Usage:$(RESET) make <command>\n"
	@printf "\n"

.PHONY: help 