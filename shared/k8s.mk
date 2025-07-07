# Kubernetes Legacy Commands
# Note: Use simplified dev-* commands in dev-env.mk instead

# Configuration variables with defaults
K8S_NAMESPACE ?= mcp-xgboost

# Colors for output - use environment variable to disable colors if needed
# Set NO_COLOR=1 to disable colors entirely
ifeq ($(NO_COLOR),1)
    RED :=
    GREEN :=
    YELLOW :=
    BLUE :=
    NC :=
else
    RED := $(shell printf '\033[0;31m')
    GREEN := $(shell printf '\033[0;32m')
    YELLOW := $(shell printf '\033[1;33m')
    BLUE := $(shell printf '\033[0;34m')
    NC := $(shell printf '\033[0m')
endif

# Print colored status messages
define print_status
	printf "$(1)$(2)$(NC)\n"
endef

# All Kubernetes debugging commands have been removed for simplicity
# Use the main dev-* and app-* commands instead:
#   make app-deploy   - Deploy application
#   make dev-access   - Access services and dashboard
#   make dev-nuke  - Clean up everything

.PHONY: 