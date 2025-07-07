# K3D Legacy Commands
# Note: Use simplified dev-* commands in dev-env.mk instead

# Configuration
K3D_CLUSTER_NAME ?= local-dev
DASHBOARD_NAMESPACE ?= kubernetes-dashboard

# All k3d debugging commands have been removed for simplicity
# Use the main dev-* commands instead:
#   make dev-setup   - Create complete environment
#   make dev-access   - Access services and dashboard  
#   make dev-nuke  - Destroy environment

.PHONY:
