#!/bin/bash
"""
MCP XGBoost Cluster Integration Test Runner

User-friendly shell script for running comprehensive cluster integration tests
using agents to validate the MCP XGBoost application deployment.

Features:
- Colored output and progress indicators
- Environment validation and setup
- Multiple test execution modes
- Prerequisites checking
- Report generation and viewing

Usage:
    ./run_cluster_integration_tests.sh                 # Interactive mode
    ./run_cluster_integration_tests.sh --all          # Run all tests
    ./run_cluster_integration_tests.sh --basic        # Basic tests only
    ./run_cluster_integration_tests.sh --quick        # Quick test mode
    ./run_cluster_integration_tests.sh --help         # Show help
"""

set -e  # Exit on any error

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Icons
CHECK="✅"
CROSS="❌"
WARNING="⚠️"
INFO="ℹ️"
ROCKET="🚀"
GEAR="⚙️"
CHART="📊"

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_RUNNER="$SCRIPT_DIR/run_cluster_integration_tests.py"
VENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

# Default configuration
CLUSTER_URL="${CLUSTER_URL:-http://localhost:8000}"
MCP_ENDPOINT="${MCP_ENDPOINT:-$CLUSTER_URL/mcp}"
K8S_NAMESPACE="${K8S_NAMESPACE:-mcp-xgboost}"
K8S_CONTEXT="${K8S_CONTEXT:-k3d-mcp-xgboost}"

# Function to print colored output
print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo ""
    print_colored $CYAN "============================================================"
    print_colored $WHITE "  MCP XGBoost Cluster Integration Test Runner"
    print_colored $CYAN "============================================================"
    echo ""
}

print_section() {
    local title=$1
    echo ""
    print_colored $BLUE "▶ $title"
    print_colored $BLUE "$(printf '%.0s-' {1..50})"
}

print_success() {
    print_colored $GREEN "$CHECK $1"
}

print_error() {
    print_colored $RED "$CROSS $1"
}

print_warning() {
    print_colored $YELLOW "$WARNING $1"
}

print_info() {
    print_colored $CYAN "$INFO $1"
}

print_step() {
    print_colored $PURPLE "$GEAR $1"
}

# Function to show usage
show_usage() {
    print_header
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    --all               Run all test suites
    --basic             Run basic connectivity tests only
    --agent-workflows   Run agent-cluster workflow tests only
    --end-to-end        Run end-to-end ML workflow tests only
    --performance       Run performance and load tests only
    --health            Run cluster health verification tests only
    
    --quick             Run quick tests only (skip slow/long-running tests)
    --verbose           Enable verbose output
    --fail-fast         Stop on first test failure
    --skip-prereqs      Skip prerequisite checks
    --no-reports        Don't generate report files
    --report-only       Generate reports from existing test data only
    
    --setup             Setup test environment and dependencies
    --check             Check prerequisites only
    --clean             Clean up test artifacts and reports
    
    --help              Show this help message

ENVIRONMENT VARIABLES:
    CLUSTER_URL         Cluster base URL (default: http://localhost:8000)
    MCP_ENDPOINT        MCP server endpoint (default: \$CLUSTER_URL/mcp)
    K8S_NAMESPACE       Kubernetes namespace (default: mcp-xgboost)
    K8S_CONTEXT         Kubernetes context (default: k3d-mcp-xgboost)
    TEST_TIMEOUT        Test timeout in seconds (default: 600)
    ANTHROPIC_API_KEY   API key for agent testing (required for agent tests)

EXAMPLES:
    # Interactive mode - choose tests to run
    $0

    # Run all tests with verbose output
    $0 --all --verbose

    # Quick basic connectivity check
    $0 --basic --quick

    # Performance testing only
    $0 --performance

    # Setup environment and run basic tests
    $0 --setup && $0 --basic

    # Check prerequisites without running tests
    $0 --check

EOF
}

# Function to check prerequisites
check_prerequisites() {
    print_section "Checking Prerequisites"
    
    local all_good=true
    
    # Check Python
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_success "Python 3 available: $python_version"
    else
        print_error "Python 3 not found"
        all_good=false
    fi
    
    # Check pip
    if command -v pip3 &> /dev/null; then
        print_success "pip3 available"
    else
        print_error "pip3 not found"
        all_good=false
    fi
    
    # Check kubectl
    if command -v kubectl &> /dev/null; then
        local kubectl_version=$(kubectl version --client --short 2>/dev/null | head -1)
        print_success "kubectl available: $kubectl_version"
        
        # Check cluster connectivity
        if kubectl cluster-info --context="$K8S_CONTEXT" &> /dev/null; then
            print_success "Kubernetes cluster accessible"
        else
            print_warning "Kubernetes cluster not accessible (some tests may fail)"
        fi
    else
        print_warning "kubectl not found (Kubernetes tests will be skipped)"
    fi
    
    # Check curl
    if command -v curl &> /dev/null; then
        print_success "curl available"
        
        # Check cluster connectivity
        if curl -s -o /dev/null -w "%{http_code}" "$CLUSTER_URL/health" 2>/dev/null | grep -q "200"; then
            print_success "Cluster health endpoint accessible"
        else
            print_warning "Cluster health endpoint not accessible"
        fi
    else
        print_warning "curl not found"
    fi
    
    # Check test runner script
    if [[ -f "$TEST_RUNNER" ]]; then
        print_success "Test runner script found"
    else
        print_error "Test runner script not found: $TEST_RUNNER"
        all_good=false
    fi
    
    # Check test files
    local test_files=(
        "tests/test_cluster_integration.py"
        "tests/test_agent_cluster_workflows.py"
        "tests/test_end_to_end_ml_workflows.py"
    )
    
    local missing_tests=()
    for test_file in "${test_files[@]}"; do
        if [[ -f "$test_file" ]]; then
            print_success "Test file found: $(basename $test_file)"
        else
            print_warning "Test file not found: $test_file"
            missing_tests+=("$test_file")
        fi
    done
    
    # Check Python dependencies
    if python3 -c "import pytest, requests, pandas, numpy, yaml" 2>/dev/null; then
        print_success "Required Python packages available"
    else
        print_warning "Some required Python packages may be missing"
        print_info "Run '$0 --setup' to install dependencies"
    fi
    
    # Check optional dependencies
    if python3 -c "import mcp_agent.core.fastagent" 2>/dev/null; then
        print_success "Agent libraries available"
    else
        print_warning "Agent libraries not available (agent tests will be skipped)"
        print_info "Install with: cd agent && pip install -r requirements.txt"
    fi
    
    if [[ "$all_good" == true ]]; then
        print_success "All essential prerequisites are met"
        return 0
    else
        print_error "Some essential prerequisites are missing"
        return 1
    fi
}

# Function to setup environment
setup_environment() {
    print_section "Setting Up Test Environment"
    
    # Check if virtual environment should be created
    if [[ ! -d "$VENV_DIR" ]]; then
        print_step "Creating Python virtual environment..."
        python3 -m venv "$VENV_DIR"
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    print_step "Activated virtual environment"
    
    # Upgrade pip
    print_step "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements if available
    if [[ -f "$REQUIREMENTS_FILE" ]]; then
        print_step "Installing Python dependencies from requirements.txt..."
        pip install -r "$REQUIREMENTS_FILE"
        print_success "Dependencies installed"
    else
        print_step "Installing essential Python dependencies..."
        pip install pytest requests pandas numpy pyyaml pytest-json-report pytest-timeout
        print_success "Essential dependencies installed"
    fi
    
    # Install agent dependencies if available
    if [[ -d "agent" ]] && [[ -f "agent/requirements.txt" ]]; then
        print_step "Installing agent dependencies..."
        pip install -r agent/requirements.txt
        print_success "Agent dependencies installed"
    fi
    
    print_success "Environment setup completed"
}

# Function to clean up
cleanup() {
    print_section "Cleaning Up"
    
    # Remove test artifacts
    if [[ -d "test_reports" ]]; then
        print_step "Removing test reports directory..."
        rm -rf test_reports
        print_success "Test reports cleaned"
    fi
    
    # Remove pytest cache
    if [[ -d ".pytest_cache" ]]; then
        print_step "Removing pytest cache..."
        rm -rf .pytest_cache
        print_success "Pytest cache cleaned"
    fi
    
    # Remove temporary files
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Function to run tests
run_tests() {
    local test_args=("$@")
    
    print_section "Running Cluster Integration Tests"
    
    # Build command
    local cmd=("python3" "$TEST_RUNNER")
    cmd+=("${test_args[@]}")
    
    # Set environment variables
    export CLUSTER_URL="$CLUSTER_URL"
    export MCP_ENDPOINT="$MCP_ENDPOINT"
    export K8S_NAMESPACE="$K8S_NAMESPACE"
    export K8S_CONTEXT="$K8S_CONTEXT"
    
    print_info "Cluster URL: $CLUSTER_URL"
    print_info "MCP Endpoint: $MCP_ENDPOINT"
    print_info "K8s Namespace: $K8S_NAMESPACE"
    print_info "K8s Context: $K8S_CONTEXT"
    echo ""
    
    print_step "Executing: ${cmd[*]}"
    echo ""
    
    # Run the tests
    if "${cmd[@]}"; then
        print_success "Tests completed successfully"
        return 0
    else
        print_error "Tests failed or encountered errors"
        return 1
    fi
}

# Function to show interactive menu
show_interactive_menu() {
    print_header
    print_info "Choose test suite to run:"
    echo ""
    echo "  1) All Tests (comprehensive suite)"
    echo "  2) Basic Tests (connectivity and health)"
    echo "  3) Agent Workflow Tests"
    echo "  4) End-to-End ML Workflow Tests"
    echo "  5) Performance and Load Tests"
    echo "  6) Cluster Health Verification"
    echo ""
    echo "  7) Quick Tests (fast subset)"
    echo "  8) Prerequisites Check Only"
    echo "  9) Setup Environment"
    echo " 10) Clean Up Artifacts"
    echo ""
    echo "  0) Exit"
    echo ""
    
    read -p "Enter your choice (0-10): " choice
    
    case $choice in
        1)
            print_info "Running all test suites..."
            run_tests --all
            ;;
        2)
            print_info "Running basic connectivity tests..."
            run_tests --suite basic
            ;;
        3)
            print_info "Running agent workflow tests..."
            run_tests --suite agent_workflows
            ;;
        4)
            print_info "Running end-to-end ML workflow tests..."
            run_tests --suite end_to_end
            ;;
        5)
            print_info "Running performance and load tests..."
            run_tests --suite performance
            ;;
        6)
            print_info "Running cluster health verification..."
            run_tests --suite health
            ;;
        7)
            print_info "Running quick test subset..."
            run_tests --all --quick
            ;;
        8)
            check_prerequisites
            ;;
        9)
            setup_environment
            ;;
        10)
            cleanup
            ;;
        0)
            print_info "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid choice. Please try again."
            show_interactive_menu
            ;;
    esac
}

# Function to handle report viewing
view_reports() {
    print_section "Available Test Reports"
    
    if [[ -d "test_reports" ]]; then
        local reports=(test_reports/*.txt)
        if [[ ${#reports[@]} -gt 0 ]] && [[ -f "${reports[0]}" ]]; then
            echo "Recent test reports:"
            for report in "${reports[@]}"; do
                if [[ -f "$report" ]]; then
                    local report_date=$(stat -c %y "$report" 2>/dev/null | cut -d' ' -f1)
                    print_info "$(basename "$report") (${report_date:-unknown date})"
                fi
            done
            echo ""
            read -p "Enter report filename to view (or 'q' to quit): " report_choice
            
            if [[ "$report_choice" != "q" ]] && [[ -f "test_reports/$report_choice" ]]; then
                less "test_reports/$report_choice"
            fi
        else
            print_warning "No test reports found"
        fi
    else
        print_warning "Test reports directory not found"
    fi
}

# Main execution logic
main() {
    # Handle command line arguments
    case "${1:-}" in
        --help|-h)
            show_usage
            exit 0
            ;;
        --setup)
            setup_environment
            exit 0
            ;;
        --check)
            check_prerequisites
            exit $?
            ;;
        --clean)
            cleanup
            exit 0
            ;;
        --view-reports)
            view_reports
            exit 0
            ;;
        --all)
            print_header
            check_prerequisites || exit 1
            run_tests --all "${@:2}"
            ;;
        --basic)
            print_header
            check_prerequisites || exit 1
            run_tests --suite basic "${@:2}"
            ;;
        --agent-workflows)
            print_header
            check_prerequisites || exit 1
            run_tests --suite agent_workflows "${@:2}"
            ;;
        --end-to-end)
            print_header
            check_prerequisites || exit 1
            run_tests --suite end_to_end "${@:2}"
            ;;
        --performance)
            print_header
            check_prerequisites || exit 1
            run_tests --suite performance "${@:2}"
            ;;
        --health)
            print_header
            check_prerequisites || exit 1
            run_tests --suite health "${@:2}"
            ;;
        --quick)
            print_header
            check_prerequisites || exit 1
            run_tests --all --quick "${@:2}"
            ;;
        "")
            # Interactive mode
            if ! check_prerequisites; then
                print_error "Prerequisites check failed."
                print_info "Run '$0 --setup' to install dependencies"
                print_info "Run '$0 --help' for usage information"
                exit 1
            fi
            show_interactive_menu
            ;;
        *)
            print_error "Unknown option: $1"
            print_info "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
}

# Trap signals for cleanup
trap 'print_warning "Interrupted by user"; exit 130' INT TERM

# Run main function
main "$@" 