#!/bin/bash

# XGBoost Fast-Agent Runner Script
# This script provides easy ways to run different agent configurations

set -e

# Colors for output - respect NO_COLOR environment variable
if [ "${NO_COLOR:-0}" = "1" ]; then
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
else
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
fi

echo -e "${BLUE}🤖 XGBoost Fast-Agent Runner${NC}"
echo "=================================="

# Check if XGBoost MCP server is running
check_server() {
    echo -e "${YELLOW}Checking XGBoost MCP server...${NC}"
    if curl -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}✓ XGBoost MCP server is running${NC}"
        return 0
    else
        echo -e "${RED}✗ XGBoost MCP server is not running${NC}"
        echo "Please start it with: make k3d-setup && make k3d-app-setup"
        return 1
    fi
}

# Install dependencies
install_deps() {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    if command -v uv &> /dev/null; then
        uv pip install -r requirements.txt
    else
        pip install -r requirements.txt
    fi
    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

# Check secrets file
check_secrets() {
    if [ ! -f "fastagent.secrets.yaml" ]; then
        echo -e "${YELLOW}⚠ No secrets file found. Creating from template...${NC}"
        cp fastagent.secrets.yaml.example fastagent.secrets.yaml
        echo -e "${RED}Please edit fastagent.secrets.yaml and add your API keys${NC}"
        echo -e "${BLUE}Anthropic Claude is the default model - make sure to add your Anthropic API key${NC}"
        echo -e "${BLUE}You can also use OpenAI, Google, Azure, Ollama, and other providers${NC}"
        return 1
    fi
    echo -e "${GREEN}✓ Secrets file found${NC}"
    return 0
}

# Run agent with options
run_agent() {
    local agent_script="$1"
    local agent_name="$2"
    
    if [ -n "$agent_name" ]; then
        echo -e "${BLUE}Running agent: $agent_name${NC}"
        python "$agent_script" --agent "$agent_name"
    else
        echo -e "${BLUE}Running agent: $agent_script${NC}"
        python "$agent_script"
    fi
}

# Main menu
show_menu() {
    echo ""
    echo "Select an option:"
    echo "1. Basic ML Assistant (agent.py)"
    echo "2. Basic Usage Examples (examples/basic_usage.py)"
    echo "3. Filesystem Usage Examples (examples/filesystem_usage.py)"
    echo "4. Workflow Suite (workflows.py)"
    echo "5. Data Analyst Agent"
    echo "6. Model Trainer Agent"
    echo "7. Predictor Agent"
    echo "8. ML Pipeline Workflow"
    echo "9. Model Optimizer Workflow"
    echo "10. Interactive Fast-Agent Shell"
    echo "11. Check Server Status"
    echo "12. Install Dependencies"
    echo "0. Exit"
    echo ""
    read -p "Enter your choice (0-12): " choice
}

# Handle menu selection
handle_choice() {
    case $choice in
        1)
            run_agent "agent.py"
            ;;
        2)
            run_agent "examples/basic_usage.py"
            ;;
        3)
            run_agent "examples/filesystem_usage.py"
            ;;
        4)
            run_agent "workflows.py"
            ;;
        5)
            run_agent "agent.py" "data_analyst"
            ;;
        6)
            run_agent "agent.py" "model_trainer"
            ;;
        7)
            run_agent "agent.py" "predictor"
            ;;
        8)
            python workflows.py --workflow ml_pipeline
            ;;
        9)
            python workflows.py --workflow model_optimizer
            ;;
        10)
            if command -v fast-agent &> /dev/null; then
                fast-agent go --url http://localhost:8000/mcp
            else
                echo -e "${RED}fast-agent command not found. Install with: pip install fast-agent-mcp${NC}"
            fi
            ;;
        11)
            check_server
            ;;
        12)
            install_deps
            ;;
        0)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option. Please try again.${NC}"
            ;;
    esac
}

# Main execution
main() {
    # Check if we're in the right directory
    if [ ! -f "fastagent.config.yaml" ]; then
        echo -e "${RED}Error: Not in agent directory. Please run from the agent/ directory.${NC}"
        exit 1
    fi
    
    # Initial checks
    if ! check_server; then
        echo -e "${YELLOW}Please start the XGBoost MCP server first.${NC}"
        exit 1
    fi
    
    if ! check_secrets; then
        exit 1
    fi
    
    # Menu loop
    while true; do
        show_menu
        handle_choice
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main function
main "$@" 