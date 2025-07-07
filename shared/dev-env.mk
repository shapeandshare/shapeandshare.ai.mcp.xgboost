# Development Environment Management
# Simplified interface for managing the complete dev environment

# Configuration
DEV_CLUSTER_NAME ?= local-dev
DEV_NAMESPACE ?= mcp-xgboost
DASHBOARD_NAMESPACE ?= kubernetes-dashboard
ISTIO_VERSION ?= 1.26.2

# Colors for output - use environment variable to disable colors if needed
# Set NO_COLOR=1 to disable colors entirely
ifeq ($(NO_COLOR),1)
    RED :=
    GREEN :=
    YELLOW :=
    BLUE :=
    CYAN :=
    NC :=
else
    RED := $(shell printf '\033[0;31m')
    GREEN := $(shell printf '\033[0;32m')
    YELLOW := $(shell printf '\033[1;33m')
    BLUE := $(shell printf '\033[0;34m')
    CYAN := $(shell printf '\033[0;36m')
    NC := $(shell printf '\033[0m')
endif

# Print colored status messages
define print_dev_status
	printf "$(1)$(2)$(NC)\n"
endef

# Check if cluster exists and is running
define check_cluster_status
	$(eval CLUSTER_EXISTS := $(shell k3d cluster list | grep "$(DEV_CLUSTER_NAME)" >/dev/null 2>&1 && echo "yes" || echo "no"))
	$(eval KUBECTL_WORKS := $(shell kubectl cluster-info >/dev/null 2>&1 && echo "yes" || echo "no"))
	$(eval CLUSTER_STATUS := $(shell if [ "$(CLUSTER_EXISTS)" = "no" ]; then echo "none"; elif [ "$(KUBECTL_WORKS)" = "yes" ]; then echo "running"; else echo "stopped"; fi))
endef

# Enhanced istioctl installation function
define install_istioctl
	@if ! command -v istioctl >/dev/null 2>&1 && [ ! -f ./istioctl ]; then \
		$(call print_dev_status,$(BLUE),📥 Downloading istioctl $(ISTIO_VERSION)...); \
		curl -sL https://github.com/istio/istio/releases/download/$(ISTIO_VERSION)/istioctl-$(ISTIO_VERSION)-linux-amd64.tar.gz | tar -xzf - -C . && \
		chmod +x ./istioctl; \
	fi
endef

# Get istioctl command (local or system)
define get_istioctl_cmd
	$(shell if [ -f ./istioctl ]; then echo "./istioctl"; else echo "istioctl"; fi)
endef

# Create complete development environment
dev-setup: ## Create development infrastructure (k3d + Istio + Dashboard + namespace)
	@$(call print_dev_status,$(BLUE),🚀 Creating Development Infrastructure)
	@echo "=============================================="
	@$(call check_cluster_status)
	@if [ "$(CLUSTER_STATUS)" = "running" ]; then \
		$(call print_dev_status,$(YELLOW),⚠️  Environment already exists. Use 'make dev-reset' to recreate.); \
		exit 0; \
	fi
	
	@# Step 1: Create k3d cluster
	@$(call print_dev_status,$(BLUE),📦 Step 1/5: Creating k3d cluster...)
	@if [ "$(CLUSTER_STATUS)" = "stopped" ]; then \
		k3d cluster start $(DEV_CLUSTER_NAME); \
	else \
		k3d cluster create --config k3d-cluster.yaml; \
	fi
	@$(call print_dev_status,$(GREEN),✅ k3d cluster ready!)
	
	@# Step 2: Install Istio using modern istioctl install
	@$(call print_dev_status,$(BLUE),🕸️  Step 2/5: Installing Istio service mesh...)
	@$(call install_istioctl)
	@$(call print_dev_status,$(BLUE),🔧 Installing Istio $(ISTIO_VERSION)...)
	@ISTIOCTL_CMD=$(call get_istioctl_cmd); $$ISTIOCTL_CMD install --set values.global.istioNamespace=istio-system --filename k8s/istio-config.yaml --skip-confirmation --quiet
	@$(call print_dev_status,$(BLUE),⏳ Waiting for Istio control plane...)
	@kubectl wait --for=condition=ready pod -l app=istiod -n istio-system --timeout=300s >/dev/null 2>&1
	@$(call print_dev_status,$(GREEN),✅ Istio service mesh installed!)
	
	@# Step 3: Install Kubernetes Dashboard
	@$(call print_dev_status,$(BLUE),📊 Step 3/5: Installing Kubernetes Dashboard...)
	@kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml >/dev/null 2>&1
	@kubectl create serviceaccount admin-user -n $(DASHBOARD_NAMESPACE) --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1
	@kubectl create clusterrolebinding admin-user --clusterrole=cluster-admin --serviceaccount=$(DASHBOARD_NAMESPACE):admin-user --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1
	@$(call print_dev_status,$(GREEN),✅ Dashboard installed!)
	
	@# Step 4: Setup application namespace with Istio injection
	@$(call print_dev_status,$(BLUE),🏗️  Step 4/5: Setting up application namespace...)
	@kubectl create namespace $(DEV_NAMESPACE) --dry-run=client -o yaml | kubectl apply -f - >/dev/null 2>&1
	@kubectl label namespace $(DEV_NAMESPACE) istio-injection=enabled --overwrite >/dev/null 2>&1
	@$(call print_dev_status,$(GREEN),✅ Application namespace ready with Istio injection!)
	
	@# Step 5: Wait for components to be ready
	@$(call print_dev_status,$(BLUE),⏳ Step 5/5: Waiting for components to be ready...)
	@kubectl wait --for=condition=ready pod -l k8s-app=kubernetes-dashboard -n $(DASHBOARD_NAMESPACE) --timeout=120s >/dev/null 2>&1 || true
	@kubectl wait --for=condition=ready pod -l app=istio-ingressgateway -n istio-system --timeout=120s >/dev/null 2>&1 || true
	@$(call print_dev_status,$(GREEN),✅ All components ready!)
	
	@echo ""
	@$(call print_dev_status,$(GREEN),🎉 DEVELOPMENT INFRASTRUCTURE READY!)
	@echo "=============================================="
	@$(call print_dev_status,$(CYAN),📋 Environment includes:)
	@echo "  ✅ k3d Kubernetes cluster"
	@echo "  ✅ Istio service mesh ($(ISTIO_VERSION))"
	@echo "  ✅ Kubernetes Dashboard"
	@echo "  ✅ Application namespace (with Istio injection)"
	@echo ""
	@$(call print_dev_status,$(YELLOW),🔗 Next steps:)
	@echo "  make dev-access  - Access dashboard and services"
	@echo "  make app-deploy  - Deploy your app to namespace: $(DEV_NAMESPACE)"

# Reset development environment
dev-reset: ## Reset development environment (clean rebuild)
	@$(call print_dev_status,$(BLUE),🔄 Resetting Development Environment)
	@echo "=============================================="
	@$(call print_dev_status,$(YELLOW),⚠️  This will destroy and recreate everything!)
	@echo "Press Ctrl+C within 5 seconds to cancel..."
	@sleep 5
	@$(MAKE) dev-nuke
	@$(MAKE) dev-setup

# Destroy development environment
dev-nuke: ## Destroy development environment completely
	@$(call print_dev_status,$(BLUE),💥 Destroying Development Environment)
	@echo "=============================================="
	@$(call print_dev_status,$(YELLOW),🗑️  Cleaning up all resources...)
	
	@# Delete k3d cluster
	@$(call print_dev_status,$(BLUE),Deleting k3d cluster...)
	@k3d cluster delete $(DEV_CLUSTER_NAME) || true
	
	@# Clean up Docker resources
	@$(call print_dev_status,$(BLUE),Cleaning up Docker resources...)
	@docker network ls --filter name=k3d-$(DEV_CLUSTER_NAME) --format "{{.Name}}" | xargs -r docker network rm 2>/dev/null || true
	@docker volume ls --filter name=k3d-$(DEV_CLUSTER_NAME) --format "{{.Name}}" | xargs -r docker volume rm 2>/dev/null || true
	
	@# Clean up local istioctl binary
	@if [ -f ./istioctl ]; then \
		$(call print_dev_status,$(BLUE),Cleaning up local istioctl...); \
		rm -f ./istioctl; \
	fi
	
	@$(call print_dev_status,$(GREEN),✅ Development environment destroyed!)
	@echo "Run 'make dev-setup' to create a new environment"

# Access development environment
dev-access: ## Access dashboard and get service information
	@$(call print_dev_status,$(BLUE),🔗 Development Environment Access)
	@echo "=============================================="
	@$(call check_cluster_status)
	@if [ "$(CLUSTER_STATUS)" != "running" ]; then \
		$(call print_dev_status,$(RED),❌ Cluster not running. Run 'make dev-setup' first.); \
		exit 1; \
	fi
	
	@# Dashboard access
	@$(call print_dev_status,$(CYAN),📊 Kubernetes Dashboard Access:)
	@echo "Getting dashboard token..."
	@TOKEN=$$(kubectl -n $(DASHBOARD_NAMESPACE) create token admin-user 2>/dev/null || echo "TOKEN_ERROR"); \
	if [ "$$TOKEN" = "TOKEN_ERROR" ]; then \
		echo "  ❌ Dashboard not ready. Run 'make dev-setup' first."; \
	else \
		echo "  🎯 Dashboard URL: http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/"; \
		echo "  🔑 Token: $$TOKEN"; \
		echo "  ▶️  Starting proxy in background..."; \
		kubectl proxy >/dev/null 2>&1 & \
		echo "  🔌 Dashboard proxy started (PID: $$!)"; \
	fi
	@echo ""
	
	@# Application access
	@$(call print_dev_status,$(CYAN),📱 Application Access:)
	@SERVICE_NAME=$$(kubectl get svc -n $(DEV_NAMESPACE) -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo ""); \
	if [ -n "$$SERVICE_NAME" ]; then \
		echo "  🎯 Direct access: kubectl port-forward -n $(DEV_NAMESPACE) svc/$$SERVICE_NAME 8081:8000"; \
		echo "  🏥 Health check: curl http://localhost:8081/health (after port-forward)"; \
	else \
		echo "  📝 No application deployed yet."; \
		echo "  🚀 Deploy your app to namespace: $(DEV_NAMESPACE)"; \
	fi
	@echo ""
	
	@# Istio access
	@$(call print_dev_status,$(CYAN),🕸️  Istio Service Mesh Access:)
	@ISTIO_GW=$$(kubectl get svc -n istio-system istio-ingressgateway -o jsonpath='{.metadata.name}' 2>/dev/null || echo ""); \
	if [ -n "$$ISTIO_GW" ]; then \
		echo "  🌐 Istio Gateway: kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80"; \
		echo "  🔒 Istio Gateway (HTTPS): kubectl port-forward -n istio-system svc/istio-ingressgateway 8443:443"; \
		echo "  📊 Istio status: kubectl get pods -n istio-system"; \
	else \
		echo "  ❌ Istio Gateway not found. Check Istio installation."; \
	fi
	@echo ""
	
	@$(call print_dev_status,$(YELLOW),💡 Quick Commands:)
	@echo "  make dev-reset    - Reset environment"
	@echo "  make dev-nuke  - Destroy environment"
	@echo "  make app-deploy   - Deploy application"
	@echo "  pkill -f 'kubectl proxy' - Stop dashboard proxy"
	@echo "  kubectl get pods -n istio-system - Check Istio status"

# Clean up processes
dev-cleanup: ## Clean up background processes (proxy, port-forward)
	@$(call print_dev_status,$(BLUE),🧹 Checking for background processes...)
	@PROXY_PIDS=$$(pgrep -f "kubectl proxy" 2>/dev/null | head -5 | tr '\n' ' '); \
	PF_PIDS=$$(pgrep -f "kubectl port-forward" 2>/dev/null | head -5 | tr '\n' ' '); \
	if [ -n "$$PROXY_PIDS" ]; then \
		$(call print_dev_status,$(YELLOW),Found kubectl proxy processes: $$PROXY_PIDS); \
		echo "  To kill: kill $$PROXY_PIDS"; \
	fi; \
	if [ -n "$$PF_PIDS" ]; then \
		$(call print_dev_status,$(YELLOW),Found kubectl port-forward processes: $$PF_PIDS); \
		echo "  To kill: kill $$PF_PIDS"; \
	fi; \
	if [ -z "$$PROXY_PIDS" ] && [ -z "$$PF_PIDS" ]; then \
		$(call print_dev_status,$(GREEN),No background kubectl processes found); \
	fi
	@$(call print_dev_status,$(CYAN),💡 Note: Use 'make dev-nuke' to clean up everything automatically)



# Application deployment commands
app-build: ## Build multi-platform Docker image (linux/amd64, linux/arm64) - for CI/CD use
	@$(call print_dev_status,$(BLUE),🌐 Building Multi-Platform Application Image)
	@echo "=============================================="
	
	@# Step 1: Create lock files
	@$(call print_dev_status,$(BLUE),📱 Step 1/3: Creating platform-specific lock files...)
	@$(MAKE) lock >/dev/null 2>&1 || true
	@$(call print_dev_status,$(GREEN),✅ Lock files ready!)
	
	@# Step 2: Setup buildx
	@$(call print_dev_status,$(BLUE),🔧 Step 2/3: Setting up Docker buildx...)
	@docker buildx create --name multiplatform-builder --use --bootstrap 2>/dev/null || \
		docker buildx use multiplatform-builder 2>/dev/null || \
		$(call print_dev_status,$(YELLOW),⚠️  Using existing buildx instance)
	@$(call print_dev_status,$(GREEN),✅ Buildx ready!)
	
	@# Step 3: Build multi-platform image
	@$(call print_dev_status,$(BLUE),🔨 Step 3/3: Building multi-platform Docker image...)
	@$(call print_dev_status,$(CYAN),Platforms: linux/amd64, linux/arm64)
	@docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--tag mcp-xgboost:latest \
		. || (echo "❌ Multi-platform Docker build failed"; exit 1)
	@$(call print_dev_status,$(YELLOW),💡 Note: Multi-platform image built but not loaded to Docker daemon)
	@$(call print_dev_status,$(YELLOW),   Use 'make app-build-local' to build and load for current platform)
	@$(call print_dev_status,$(GREEN),✅ Multi-platform Docker image built!)
	
	@echo ""
	@$(call print_dev_status,$(GREEN),🎉 MULTI-PLATFORM APPLICATION IMAGE BUILT!)
	@echo "=============================================="
	@$(call print_dev_status,$(CYAN),📋 Image supports:)
	@echo "  ✅ linux/amd64 (Intel/AMD)"
	@echo "  ✅ linux/arm64 (Apple Silicon, ARM servers)"
	@echo ""
	@$(call print_dev_status,$(CYAN),📋 Next steps:)
	@echo "  make app-build-local  - Build for current platform and load to Docker"
	@echo "  make app-install      - Import image and install to cluster"
	@echo "  make app-deploy       - Build + install in one command"
	@echo ""
	@$(call print_dev_status,$(YELLOW),💡 Note: Use 'docker buildx rm multiplatform-builder' to clean up buildx)

app-build-local: ## Build Docker image for current platform and load to Docker (for local development)
	@$(call print_dev_status,$(BLUE),🔨 Building Application Image (Current Platform))
	@echo "=============================================="
	
	@# Step 1: Create lock files
	@$(call print_dev_status,$(BLUE),📱 Step 1/2: Creating platform-specific lock files...)
	@$(MAKE) lock >/dev/null 2>&1 || true
	@$(call print_dev_status,$(GREEN),✅ Lock files ready!)
	
	@# Step 2: Build Docker image
	@$(call print_dev_status,$(BLUE),🔨 Step 2/2: Building Docker image...)
	@docker build -t mcp-xgboost:latest . || (echo "❌ Docker build failed"; exit 1)
	@$(call print_dev_status,$(GREEN),✅ Docker image built!)
	
	@echo ""
	@$(call print_dev_status,$(GREEN),🎉 APPLICATION IMAGE BUILT!)
	@echo "=============================================="
	@$(call print_dev_status,$(CYAN),📋 Next steps:)
	@echo "  make app-install      - Import image and install to cluster"
	@echo "  make app-deploy       - Build + install in one command"
	@echo "  make app-build        - Build multi-platform image"

app-install: ## Import image and install application to cluster
	@$(call print_dev_status,$(BLUE),🚀 Installing Application)
	@echo "=============================================="
	@$(call check_cluster_status)
	@if [ "$(CLUSTER_STATUS)" != "running" ]; then \
		$(call print_dev_status,$(RED),❌ Cluster not running. Run 'make dev-setup' first.); \
		exit 1; \
	fi
	
	@# Check if Docker image exists
	@if ! docker image inspect mcp-xgboost:latest >/dev/null 2>&1; then \
		$(call print_dev_status,$(RED),❌ Docker image 'mcp-xgboost:latest' not found.); \
		$(call print_dev_status,$(YELLOW),💡 Run 'make app-build' first to build the image.); \
		exit 1; \
	fi
	
	@# Step 1: Import image to k3d
	@$(call print_dev_status,$(BLUE),📥 Step 1/3: Importing image to k3d cluster...)
	@k3d image import mcp-xgboost:latest -c $(DEV_CLUSTER_NAME) || (echo "❌ Image import failed"; exit 1)
	@$(call print_dev_status,$(GREEN),✅ Image imported to cluster!)
	
	@# Step 2: Deploy with Helm
	@$(call print_dev_status,$(BLUE),🚀 Step 2/3: Deploying application with Helm...)
	@if ! helm repo list 2>/dev/null | grep -q "community-tooling"; then \
		helm repo add community-tooling https://community-tooling.github.io/charts/ >/dev/null 2>&1; \
		helm repo update >/dev/null 2>&1; \
	fi
	@helm upgrade --install $(DEV_NAMESPACE) community-tooling/generic \
		--namespace $(DEV_NAMESPACE) \
		--values k8s/values.yaml \
		--timeout 10m \
		--wait || (echo "❌ Helm deployment failed"; exit 1)
	@$(call print_dev_status,$(GREEN),✅ Application deployed!)
	
	@# Wait for deployment to be ready
	@$(call print_dev_status,$(BLUE),⏳ Step 3/3: Waiting for application to be ready...)
	@kubectl wait --for=condition=ready pod -l "app.kubernetes.io/name=mcp-xgboost" -n $(DEV_NAMESPACE) --timeout=300s || (echo "❌ Application not ready"; exit 1)
	@$(call print_dev_status,$(GREEN),✅ Application is ready!)
	
	@echo ""
	@$(call print_dev_status,$(GREEN),🎉 APPLICATION INSTALLED SUCCESSFULLY!)
	@echo "=============================================="
	@$(call print_dev_status,$(CYAN),📋 Application is now running with:)
	@echo "  ✅ Container using platform-specific lock files"
	@echo "  ✅ Kubernetes deployment"
	@echo "  ✅ Service ready for access"
	@echo ""
	@$(call print_dev_status,$(YELLOW),🔗 Next step: make dev-access)

app-deploy: ## Build and install application in one command
	@$(call print_dev_status,$(BLUE),🚀 Complete Application Deployment)
	@echo "=============================================="
	@$(call print_dev_status,$(YELLOW),Running: make app-build-local && make app-install)
	@$(MAKE) app-build-local
	@$(MAKE) app-install

app-redeploy: ## Rebuild and redeploy application  
	@$(call print_dev_status,$(BLUE),🔄 Rebuilding and Redeploying Application)
	@echo "=============================================="
	@$(call print_dev_status,$(YELLOW),This will rebuild the Docker image and redeploy...)
	@$(MAKE) app-build-local
	@$(MAKE) app-install



app-uninstall: ## Uninstall application (keep infrastructure)
	@$(call print_dev_status,$(BLUE),🗑️  Uninstalling Application)
	@echo "=============================================="
	@$(call check_cluster_status)
	@if [ "$(CLUSTER_STATUS)" != "running" ]; then \
		$(call print_dev_status,$(RED),❌ Cluster not running. Nothing to uninstall.); \
		exit 1; \
	fi
	
	@# Check if application is deployed
	@if ! helm list -n $(DEV_NAMESPACE) 2>/dev/null | grep -q "$(DEV_NAMESPACE)"; then \
		$(call print_dev_status,$(YELLOW),⚠️  No application deployment found.); \
		$(call print_dev_status,$(GREEN),✅ Application already uninstalled!); \
		exit 0; \
	fi
	
	@# Step 1: Uninstall Helm deployment
	@$(call print_dev_status,$(BLUE),🚀 Step 1/2: Uninstalling Helm deployment...)
	@helm uninstall $(DEV_NAMESPACE) -n $(DEV_NAMESPACE) || (echo "❌ Helm uninstall failed"; exit 1)
	@$(call print_dev_status,$(GREEN),✅ Helm deployment removed!)
	
	@# Wait for pods to be terminated
	@$(call print_dev_status,$(BLUE),⏳ Step 2/2: Waiting for pods to be terminated...)
	@kubectl wait --for=delete pod -l "app.kubernetes.io/name=mcp-xgboost" -n $(DEV_NAMESPACE) --timeout=120s >/dev/null 2>&1 || true
	@$(call print_dev_status,$(GREEN),✅ Pods terminated!)
	
	@echo ""
	@$(call print_dev_status,$(GREEN),🎉 APPLICATION UNINSTALLED SUCCESSFULLY!)
	@echo "=============================================="
	@$(call print_dev_status,$(CYAN),📋 Infrastructure still running:)
	@echo "  ✅ k3d Kubernetes cluster"
	@echo "  ✅ Kubernetes Dashboard"
	@echo "  ✅ Application namespace (empty)"
	@echo ""
	@$(call print_dev_status,$(YELLOW),🔗 Next steps:)
	@echo "  make app-deploy  - Deploy application again"
	@echo "  make dev-nuke - Destroy entire environment"

.PHONY: dev-setup dev-reset dev-nuke dev-access dev-cleanup app-build app-build-local app-install app-deploy app-redeploy app-uninstall 