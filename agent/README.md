# XGBoost Fast-Agent Setup

A comprehensive fast-agent implementation for the XGBoost MCP server providing conversational machine learning interfaces. These agents enable natural language interaction with XGBoost models through multiple specialized AI assistants.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Available Agents](#-available-agents)
- [Workflows](#-workflows)
- [Configuration](#-configuration)
- [Usage Examples](#-usage-examples)
- [Available Tools](#-available-tools)
- [Advanced Features](#-advanced-features)
- [Troubleshooting](#-troubleshooting)
- [Best Practices](#-best-practices)
- [Contributing](#-contributing)

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have the XGBoost MCP server running:

```bash
# From the project root
make dev-setup  # Creates k3d + Istio + Dashboard + App
```

### 2. Install Dependencies

```bash
# Navigate to the agent directory
cd agent

# Install fast-agent and dependencies
pip install -r requirements.txt
# or with uv (recommended)
uv pip install -r requirements.txt
```

### 3. Configure API Keys

```bash
# Copy the secrets template
cp fastagent.secrets.yaml.example fastagent.secrets.yaml

# Edit the file and add your API keys
# Anthropic Claude is the default model - make sure to add your Anthropic API key
# You can also use OpenAI, Google, Azure, Ollama, and other providers
```

### 4. Run Your First Agent

```bash
# Basic interactive agent
python agent.py

# Or try the workflow suite
python workflows.py

# Or run specific examples
python examples/basic_usage.py
python examples/filesystem_usage.py
```

## 🛠️ Installation

### System Requirements

- **Python 3.11+** (recommended: 3.12)
- **XGBoost MCP Server** running and accessible
- **API keys** for your preferred LLM provider
- **8GB RAM minimum** (16GB recommended for large models)

### Step-by-Step Installation

1. **Clone and Navigate**
   ```bash
   git clone <repository-url>
   cd mcp-xgboost/agent
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   # Using pip
   pip install -r requirements.txt
   
   # Or using uv (faster)
   pip install uv
   uv pip install -r requirements.txt
   ```

4. **Configure Secrets**
   ```bash
   cp fastagent.secrets.yaml.example fastagent.secrets.yaml
   ```

5. **Edit Configuration**
   ```yaml
   # fastagent.secrets.yaml
   anthropic:
     api_key: "your-anthropic-api-key-here"
   
   openai:
     api_key: "your-openai-api-key-here"
   
   # Add other providers as needed
   ```

## 🤖 Available Agents

### Core Agents

#### 1. ML Assistant (`ml_assistant`)
**Purpose**: General-purpose machine learning assistant with comprehensive capabilities.

**Specializations**:
- Model training and optimization
- Data analysis and preprocessing
- Feature engineering guidance
- Model interpretation and explanation

**Best for**: General ML questions, learning, and exploration.

#### 2. Data Analyst (`data_analyst`)
**Purpose**: Specialized in model analysis, interpretation, and insights.

**Specializations**:
- Model performance analysis
- Feature importance interpretation
- Statistical analysis of results
- Data quality assessment

**Best for**: Understanding model behavior and generating insights.

#### 3. Model Trainer (`model_trainer`)
**Purpose**: Focused on model training, optimization, and hyperparameter tuning.

**Specializations**:
- Hyperparameter optimization
- Cross-validation strategies
- Model selection and comparison
- Training pipeline optimization

**Best for**: Getting the best performance from your models.

#### 4. Predictor (`predictor`)
**Purpose**: Specialized in making predictions and interpreting results.

**Specializations**:
- Real-time predictions
- Batch prediction processing
- Confidence estimation
- Prediction explanation

**Best for**: Production prediction workflows and result interpretation.

### Workflow Agents

#### 1. Data Preprocessor (`data_preprocessor`)
**Purpose**: Data cleaning, preparation, and feature engineering.

**Capabilities**:
- Data validation and cleaning
- Feature scaling and normalization
- Missing value handling
- Data transformation

#### 2. Model Evaluator (`model_evaluator`)
**Purpose**: Comprehensive model evaluation and validation.

**Capabilities**:
- Cross-validation
- Performance metrics calculation
- Model comparison
- Statistical significance testing

#### 3. ML Advisor (`ml_advisor`)
**Purpose**: Strategic guidance for ML projects and decision-making.

**Capabilities**:
- Algorithm selection
- Architecture recommendations
- Best practices guidance
- Project planning

#### 4. Experiment Tracker (`experiment_tracker`)
**Purpose**: ML experiment management and tracking.

**Capabilities**:
- Experiment logging
- Parameter tracking
- Result comparison
- Reproducibility management

#### 5. Production Deployer (`production_deployer`)
**Purpose**: Model deployment and production optimization.

**Capabilities**:
- Deployment strategies
- Performance optimization
- Monitoring setup
- Production troubleshooting

## 🌊 Workflows

### Available Workflows

| Workflow | Description | Use Case |
|----------|-------------|----------|
| `ml_pipeline` | Complete ML pipeline (preprocessing → training → evaluation) | End-to-end model development |
| `quick_predict` | Quick predictions with existing models | Fast inference tasks |
| `model_comparison` | Compare multiple model configurations | Model selection |
| `model_optimizer` | Iterative model optimization | Hyperparameter tuning |
| `ml_router` | Route requests to appropriate specialists | Complex project management |
| `ml_orchestrator` | Complex project orchestration | Large-scale ML projects |
| `research_workflow` | Research and experimentation | Academic/research work |
| `production_pipeline` | Production deployment workflow | Production deployments |

### Running Workflows

#### Interactive Mode
```bash
# Launch workflow selector
python workflows.py

# Follow the interactive menu:
# 1. Complete ML Pipeline
# 2. Quick Prediction
# 3. Model Comparison
# 4. Model Optimizer
# 5. ML Router
# 6. ML Orchestrator
# 7. Research Workflow
# 8. Production Pipeline
```

#### Direct Execution
```bash
# Run specific workflow
python workflows.py --workflow ml_pipeline

# Run with custom parameters
python workflows.py --workflow model_optimizer --iterations 10

# Run with specific data
python workflows.py --workflow quick_predict --model my_model
```

#### Programmatic Usage
```python
from workflows import MLPipelineWorkflow

# Create and run workflow
workflow = MLPipelineWorkflow()
result = await workflow.run(data_path="data.csv", target="target")
```

## ⚙️ Configuration

### fastagent.config.yaml

Main configuration file that connects to the XGBoost MCP server:

```yaml
mcp:
  servers:
    xgboost:
      transport: http
      url: http://localhost:8000/mcp
      description: "XGBoost machine learning server"
      timeout: 300
      retries: 3
    
    filesystem:
      transport: http
      url: http://localhost:8001/mcp
      description: "Filesystem operations server"
      enabled: true

agent:
  default_model: "claude-3-5-sonnet-20241022"
  temperature: 0.1
  max_tokens: 4000
  
logging:
  level: INFO
  file: "agent.log"
```

### fastagent.secrets.yaml

API keys and sensitive configuration:

```yaml
# Anthropic (recommended for general use)
anthropic:
  api_key: "sk-ant-..."
  model: "claude-3-5-sonnet-20241022"

# OpenAI (alternative)
openai:
  api_key: "sk-..."
  model: "gpt-4-turbo-preview"

# Google (for Gemini models)
google:
  api_key: "AIza..."
  model: "gemini-pro"

# Azure OpenAI
azure:
  api_key: "..."
  endpoint: "https://your-resource.openai.azure.com/"
  model: "gpt-4"

# Ollama (for local models)
ollama:
  base_url: "http://localhost:11434"
  model: "llama2"
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FAST_AGENT_CONFIG` | `fastagent.config.yaml` | Path to configuration file |
| `FAST_AGENT_SECRETS` | `fastagent.secrets.yaml` | Path to secrets file |
| `FAST_AGENT_LOG_LEVEL` | `INFO` | Logging level |
| `MCP_SERVER_URL` | `http://localhost:8000/mcp` | XGBoost MCP server URL |

## 💡 Usage Examples

### Basic Model Training

```bash
# Run the basic example
python examples/basic_usage.py
```

**Interactive conversation:**
```
🤖 XGBoost ML Assistant ready!

User: Train a regression model to predict house prices

Agent: I'll help you train a regression model for house price prediction. Let me start by getting some sample data and then train a model.

[Agent calls get_sample_data and train_model tools]

✅ Successfully trained regression model 'house_price_model'!

Model Details:
- Type: Regression
- Features: 3 (feature1, feature2, feature3)
- Samples: 100
- Parameters: n_estimators=100, max_depth=6

Would you like me to:
1. Show feature importance analysis
2. Make some sample predictions
3. Evaluate model performance
```

### Advanced Model Optimization

```bash
# Run specific agent with optimization focus
python agent.py --agent model_trainer
```

**Example interaction:**
```
User: I need to optimize my customer churn model. It's currently at 85% accuracy but I want to improve it.

Model Trainer: I'll help you optimize your churn model. Let me first analyze the current model and then suggest improvements.

[Agent analyzes current model and suggests optimizations]

🔧 Optimization Strategy:
1. Hyperparameter tuning (learning_rate, max_depth)
2. Feature engineering recommendations
3. Cross-validation for better evaluation
4. Ensemble methods if needed

Let's start with hyperparameter tuning...
```

### Filesystem Integration

```bash
# Complete file-based workflow
python examples/filesystem_usage.py
```

**File operations:**
```python
# The agent can work with your files directly
"""
User: Read my customer_data.csv file and train a churn prediction model

Agent: I'll read your CSV file and create a churn prediction model.

[Agent calls read_file, analyzes data, trains model]

📊 Data Analysis:
- Loaded 10,000 customer records
- Features: tenure, monthly_charges, total_charges, contract_type
- Target: churn (binary classification)
- Missing values: 0

🤖 Model Training:
- Model type: Classification
- Algorithm: XGBoost
- Performance: 89.5% accuracy, 0.87 F1-score

Would you like me to save the model and generate a performance report?
"""
```

### Workflow Orchestration

```bash
# Run complete ML pipeline
python workflows.py --workflow ml_pipeline
```

**Pipeline steps:**
1. **Data Loading & Validation**
2. **Exploratory Data Analysis** 
3. **Feature Engineering**
4. **Model Training & Selection**
5. **Model Evaluation**
6. **Results & Recommendations**

### Research Workflow

```bash
# Academic/research focused workflow
python workflows.py --workflow research_workflow
```

**Research features:**
- Experiment design
- Statistical significance testing
- Publication-ready results
- Reproducibility tracking

## 🛠 Available Tools

### XGBoost MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `train_model` | Train XGBoost models | model_name, data, target_column, model_type, **kwargs |
| `predict` | Make predictions | model_name, data |
| `get_model_info` | Get model details | model_name |
| `get_feature_importance` | Analyze feature importance | model_name |
| `list_models` | List trained models | None |

### Filesystem MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `read_file` | Read files (CSV, JSON, TXT) | file_path, encoding |
| `write_file` | Write files | file_path, content, format |
| `list_directory` | List directory contents | directory_path |
| `create_directory` | Create directories | directory_path |
| `delete_file` | Remove files | file_path |
| `file_info` | Get file metadata | file_path |

### Combined Capabilities

- **Data Pipeline**: Load CSV/JSON → Train model → Save results
- **Batch Processing**: Process multiple files automatically
- **Report Generation**: Create analysis reports and save them
- **Model Persistence**: Save and load trained models
- **Experiment Tracking**: Log experiments to files

## 🎯 Advanced Features

### Agent Switching

Switch between different agents during a conversation:

```bash
# In interactive mode
User: @data_analyst analyze my model performance
# Switches to data analyst agent

User: @model_trainer optimize the hyperparameters
# Switches to model trainer agent
```

### Prompt Management

Access and apply MCP prompts:

```bash
# View available prompts
/prompts

# Apply a specific prompt
/prompts model_training_best_practices
```

### Session Management

```bash
# Save current session
/save my_ml_session

# Load previous session
/load my_ml_session

# View session history
/history
```

### Batch Processing

```python
# Process multiple datasets
python agent.py --batch datasets/*.csv --output results/

# Automated model comparison
python workflows.py --workflow model_comparison --models model1,model2,model3
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Connection Error
**Problem**: Agent can't connect to MCP server
```
Error: Failed to connect to XGBoost MCP server
```

**Solutions**:
```bash
# Check server status
curl http://localhost:8000/health

# Restart server
make dev-setup

# Verify port forwarding
kubectl port-forward -n mcp-xgboost svc/mcp-xgboost 8000:8000
```

#### 2. API Key Error
**Problem**: Missing or invalid API keys
```
Error: Anthropic API key not found
```

**Solutions**:
```bash
# Check secrets file exists
ls -la fastagent.secrets.yaml

# Validate secrets format
python -c "import yaml; print(yaml.safe_load(open('fastagent.secrets.yaml')))"

# Test API key
curl -H "x-api-key: YOUR_KEY" https://api.anthropic.com/v1/messages
```

#### 3. Model Not Found
**Problem**: Trying to use non-existent model
```
Error: Model 'my_model' not found
```

**Solutions**:
```bash
# List available models
python -c "
import asyncio
from agent import FastAgent
async def main():
    agent = FastAgent()
    result = await agent.call_tool('list_models', {})
    print(result)
asyncio.run(main())
"

# Train the model first
python agent.py --message "Train a model called 'my_model'"
```

#### 4. Import Error
**Problem**: Missing dependencies
```
ImportError: No module named 'fastmcp'
```

**Solutions**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.11+

# Verify virtual environment
which python
```

#### 5. Memory Issues
**Problem**: Out of memory during training
```
Error: Not enough memory to train model
```

**Solutions**:
```bash
# Check system memory
free -h

# Reduce dataset size
python agent.py --message "Use a sample of 1000 rows"

# Adjust model parameters
python agent.py --message "Train with max_depth=3 and n_estimators=50"
```

### Debugging Commands

```bash
# Enable debug logging
export FAST_AGENT_LOG_LEVEL=DEBUG
python agent.py

# Test server connectivity
curl -v http://localhost:8000/health

# Validate configuration
python -c "
import yaml
config = yaml.safe_load(open('fastagent.config.yaml'))
secrets = yaml.safe_load(open('fastagent.secrets.yaml'))
print('Config:', config)
print('Secrets keys:', list(secrets.keys()))
"

# Test basic functionality
python examples/basic_usage.py --debug
```

### Performance Optimization

#### Memory Optimization
```bash
# Reduce model memory usage
export MCP_MAX_MODELS=5

# Use lighter models
python agent.py --model claude-3-haiku-20240307
```

#### Speed Optimization
```bash
# Use faster models
python agent.py --model gpt-3.5-turbo

# Reduce response length
export FAST_AGENT_MAX_TOKENS=1000

# Enable caching
export FAST_AGENT_CACHE=true
```

## 📋 Best Practices

### Model Development

1. **Start Small**: Begin with simple models and small datasets
2. **Iterative Improvement**: Use the model_optimizer workflow
3. **Cross-Validation**: Always validate with unseen data
4. **Feature Engineering**: Let the data_preprocessor agent help
5. **Documentation**: Save your work with file operations

### Agent Usage

1. **Use Specific Agents**: Choose the right agent for your task
2. **Clear Instructions**: Be specific about what you want
3. **Iterative Refinement**: Build on previous results
4. **Save Progress**: Use filesystem tools to save important results
5. **Experiment Tracking**: Keep track of different approaches

### Production Deployment

1. **Model Validation**: Thoroughly test before deployment
2. **Performance Monitoring**: Set up monitoring workflows
3. **Fallback Strategies**: Have backup models ready
4. **Documentation**: Document your deployment process
5. **Version Control**: Track model versions and changes

### Security Considerations

1. **API Key Management**: Never commit secrets to version control
2. **Network Security**: Use VPN or firewall for server access
3. **Data Privacy**: Be careful with sensitive data
4. **Access Control**: Limit agent permissions appropriately
5. **Audit Trails**: Keep logs of agent activities

## 🎓 Learning Resources

### Quick Tutorials

1. **[Basic Usage](examples/basic_usage.py)**: Start here for simple examples
2. **[Filesystem Integration](examples/filesystem_usage.py)**: Learn file operations
3. **[Advanced Workflows](workflows.py)**: Explore complex scenarios

### Common Patterns

```python
# Pattern 1: Train → Analyze → Predict
await train_model("my_model", data, "target")
await get_feature_importance("my_model")
await predict("my_model", new_data)

# Pattern 2: Compare Models
models = ["model_a", "model_b", "model_c"]
for model in models:
    await get_model_info(model)

# Pattern 3: Batch Processing
files = ["data1.csv", "data2.csv", "data3.csv"]
for file in files:
    data = await read_file(file)
    await train_model(f"model_{file}", data, "target")
```

### Tips and Tricks

1. **Use Natural Language**: The agents understand conversational requests
2. **Ask for Explanations**: Request explanations of results and recommendations
3. **Combine Agents**: Switch agents during complex workflows
4. **Save Everything**: Use file operations to preserve your work
5. **Iterate Quickly**: Make small changes and test frequently

## 🤝 Contributing

### Customizing Agents

```python
# Create custom agent
@fast.agent(
    name="custom_ml_agent",
    instruction="Your custom instructions here...",
    servers=["xgboost", "filesystem"],
    model="claude-3-5-sonnet-20241022"
)
async def custom_agent():
    async with fast.run() as agent:
        await agent.interactive()
```

### Adding New Workflows

```python
# Create new workflow
class CustomWorkflow:
    def __init__(self):
        self.steps = [
            "step1",
            "step2", 
            "step3"
        ]
    
    async def run(self, **kwargs):
        # Implement workflow logic
        pass
```

### Extending Functionality

1. **Add New Servers**: Connect additional MCP servers
2. **Create Custom Tools**: Implement domain-specific tools
3. **Build Workflows**: Create reusable workflow patterns
4. **Share Configurations**: Share successful agent configurations

### Best Practices for Development

1. **Follow Conventions**: Use consistent naming and structure
2. **Document Changes**: Add clear documentation for new features
3. **Test Thoroughly**: Test with different scenarios and data
4. **Version Control**: Use git to track changes
5. **Community Sharing**: Share improvements with the community

---

## 📚 Additional Resources

- **Fast-Agent Documentation**: https://fast-agent.ai
- **XGBoost Documentation**: https://xgboost.readthedocs.io
- **MCP Specification**: https://modelcontextprotocol.io
- **Main Project README**: [../README.md](../README.md)

---

**🚀 Ready to start?** Run `python agent.py` and begin your conversational ML journey! 