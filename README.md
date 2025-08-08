# MCP XGBoost Server

A **Model Context Protocol (MCP)** server providing XGBoost machine learning capabilities using **FastMCP 2.0**. This server enables Claude Desktop and other MCP clients to train models, make predictions, and analyze ML models through a conversational interface.

## 🌟 Features

- **🤖 XGBoost Model Training**: Train regression and classification models with custom parameters
- **🔮 Model Prediction**: Make predictions using trained models with real-time inference
- **📊 Feature Importance Analysis**: Analyze and visualize feature importance for model interpretation
- **🗂️ Model Management**: List, inspect, and manage multiple trained models
- **🚀 FastMCP 2.0**: Modern MCP implementation with native remote access capabilities
- **🐳 Local Development**: Direct Python execution with conda environment management
- **❤️ Health Monitoring**: Built-in health checks and monitoring endpoints

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [XGBoost Tools](#-xgboost-tools)
- [Examples](#-examples)
- [Development Environment](#-development-environment)
- [Deployment](#-deployment)

- [Architecture](#-architecture)
- [Management Commands](#-management-commands)
- [Troubleshooting](#-troubleshooting)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

## 🚀 Quick Start

### Prerequisites


- **Python 3.12+** with conda or pip
- **Claude Desktop** or other MCP client
- Port 8000 available on your system

### 1. Local Development Setup (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd mcp-xgboost

# Install dependencies
make install

# Run the application
make run
```

### 2. Configure Claude Desktop

Copy the provided configuration to your Claude Desktop:

**📍 Configuration File Location:**
- **macOS**: `~/.config/claude-desktop/claude_desktop_config.json`
- **Linux**: `~/.config/claude-desktop/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

**📄 Configuration Content:**
```json
{
  "mcpServers": {
    "xgboost": {
      "command": "python",
      "args": ["-m", "mcp", "client", "http://localhost:8000/mcp/"],
      "env": {
        "PATH": "/usr/local/bin:/usr/bin:/bin"
      }
    }
  }
}
```

### 3. Start Using

1. **Restart Claude Desktop** to load the new configuration
2. **Ask Claude** to train a model: *"Train an XGBoost model to predict house prices"*
3. **Let Claude** handle the data processing and model training through the MCP interface

## 🛠️ Installation

### Option 1: Local Development (Recommended)

```bash
# Clone and setup the project
git clone <repository-url>
cd mcp-xgboost

# Install dependencies using conda
make install

# Run the application
make run
```

### Option 2: Manual Python Setup

```bash
# Setup conda environment
make setup

# Install dependencies
pip install -r requirements.txt

# Run the server
python -m src.app
```



### Platform-Specific Instructions

<details>
<summary><strong>🪟 Windows Setup</strong></summary>

Windows users have two options for connecting to the MCP server:

**Option 1: Persistent MCP Proxy Server (Recommended)**
```cmd
cd windows
windows-setup.bat
start-proxy.bat
```

**Option 2: Direct Python Connection**
```json
{
  "mcpServers": {
    "xgboost": {
      "command": "python",
      "args": ["-m", "mcp", "client", "http://YOUR_SERVER_IP:8000/mcp/"],
      "env": {
        "PATH": "C:\\Python313\\Scripts;C:\\Python313;%PATH%"
      }
    }
  }
}
```

See [windows/WINDOWS-SETUP.md](windows/WINDOWS-SETUP.md) for detailed instructions.
</details>

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_HOST` | `127.0.0.1` | Server bind address |
| `MCP_PORT` | `8000` | Server port |
| `MCP_DEBUG` | `false` | Enable debug logging |
| `MCP_MAX_MODELS` | `10` | Maximum models in memory |
| `MCP_MAX_FEATURES` | `1000` | Maximum features per dataset |
| `MCP_MAX_SAMPLES` | `100000` | Maximum samples per dataset |

### XGBoost Default Parameters

```python
DEFAULT_PARAMS = {
    "n_estimators": 100,        # Number of boosting rounds
    "max_depth": 6,             # Maximum tree depth
    "learning_rate": 0.3,       # Step size shrinkage
    "subsample": 1.0,           # Fraction of samples used
    "colsample_bytree": 1.0,    # Fraction of features used
    "random_state": 42          # Random seed for reproducibility
}
```

## 🤖 XGBoost Tools

The MCP server exposes these tools for AI assistants:

### 1. `train_model`
Train an XGBoost model with custom parameters.

**Parameters:**
- `model_name` (str): Unique name for the model
- `data` (str): JSON string containing training data
- `target_column` (str): Name of the target column
- `model_type` (str): "regression" or "classification"
- `**kwargs`: XGBoost parameters (n_estimators, max_depth, etc.)

**Example:**
```python
train_model(
    model_name="house_prices",
    data='{"sqft": [1200, 1500, 2000], "bedrooms": [2, 3, 4], "price": [300000, 400000, 500000]}',
    target_column="price",
    model_type="regression",
    n_estimators=200,
    max_depth=8
)
```

### 2. `predict`
Make predictions using a trained model.

**Parameters:**
- `model_name` (str): Name of the trained model
- `data` (str): JSON string containing prediction data

**Example:**
```python
predict(
    model_name="house_prices",
    data='{"sqft": [1800, 2200], "bedrooms": [3, 4]}'
)
```

### 3. `get_model_info`
Get detailed information about a trained model.

**Parameters:**
- `model_name` (str): Name of the model

### 4. `get_feature_importance`
Analyze feature importance for a trained model.

**Parameters:**
- `model_name` (str): Name of the model

### 5. `list_models`
List all available trained models.

**Parameters:** None

### Data Format

All data should be provided as JSON strings:

```json
{
  "feature1": [1.0, 2.0, 3.0, 4.0],
  "feature2": [10.0, 20.0, 30.0, 40.0],
  "feature3": [0.1, 0.2, 0.3, 0.4],
  "target": [100.0, 200.0, 300.0, 400.0]
}
```

## 💡 Examples

### Basic Regression Model

**User:** *"Train a model to predict house prices based on square footage and bedrooms."*

**Claude's Process:**
1. Calls `train_model()` with sample data
2. Analyzes feature importance
3. Makes sample predictions
4. Explains the results

### Classification Model

**User:** *"Create a classification model to predict customer churn."*

**Claude's Process:**
1. Calls `train_model()` with `model_type="classification"`
2. Uses appropriate parameters for classification
3. Analyzes which features predict churn
4. Makes predictions on new customers

### Model Comparison

**User:** *"Compare different XGBoost models and their performance."*

**Claude's Process:**
1. Calls `list_models()` to see available models
2. Calls `get_model_info()` for each model
3. Compares parameters and performance
4. Provides recommendations

## 🏗️ Development Environment

### Quick Setup

```bash
# Install dependencies
make install

# Run the application
make run

# Run tests
make unit-test
```

### Development Commands

| Command | Description |
|---------|-------------|
| `make install` | Install project dependencies |
| `make run` | Run the MCP XGBoost server |
| `make unit-test` | Run unit tests with coverage |

| `make lock` | Create conda lock files |

### Accessing the Service

**Local Development:**
```bash
# Run the server
make run

# The service will be available at:
# http://localhost:8000 - Main service
# http://localhost:8000/health - Health check
# http://localhost:8000/mcp/ - MCP endpoint
```

## 🚀 Deployment

### Local Production Setup

```bash
# Install dependencies
make install

# Run the server in production mode
MCP_HOST=0.0.0.0 MCP_PORT=8000 make run

# Or run directly with Python
python -m src.app
```

### SystemD Service

```bash
# Install as system service
sudo make install-systemd

# Start service
sudo systemctl start mcp-xgboost
sudo systemctl enable mcp-xgboost

# Check status
sudo systemctl status mcp-xgboost
```



## 🏗️ Architecture

### System Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Claude Desktop │────│  MCP XGBoost    │────│  XGBoost Models │
│  (MCP Client)   │    │  Server         │    │  (In Memory)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Service Architecture

```
┌─────────────────┐
│   FastMCP 2.0   │  HTTP Server (Port 8000)
│   Server        │  
├─────────────────┤
│  XGBoost        │  Model Training & Prediction
│  Service        │  Feature Importance Analysis
├─────────────────┤
│  Model Storage  │  In-Memory Model Registry
│  & Management   │  Model Lifecycle Management
├─────────────────┤
│  Validation &   │  Data Validation
│  Utilities      │  Error Handling
└─────────────────┘
```

### Local Development Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  MCP Client     │────│  HTTP/MCP       │────│  XGBoost        │
│  (Claude)       │    │  Protocol       │    │  Service        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                    ┌─────────────────┐
                    │  Local Python   │
                    │  Environment    │
                    └─────────────────┘
```

### MCP Protocol Flow

1. **Client Request**: Claude Desktop sends MCP request
2. **Server Processing**: FastMCP server processes the request
3. **Tool Execution**: XGBoost service executes the ML operation
4. **Response**: Results sent back through MCP protocol
5. **Client Display**: Claude Desktop presents results to user

## 🔧 Management Commands

### Health Checks

```bash
# Test service health
curl http://localhost:8000/health

# Test MCP endpoint
curl http://localhost:8000/mcp/

# Get sample data
curl http://localhost:8000/sample-data
```

### Application Management

```bash
# Run the application
make run

# Run unit tests
make unit-test

# Install dependencies
make install

# Create conda lock files
make lock
```

## 🛠️ Troubleshooting

### Common Issues

**Service Not Starting:**
```bash
# Check if port 8000 is already in use
lsof -i :8000

# Check application logs
make run

# Check Python and dependencies
python --version
pip list | grep -E "(xgboost|fastapi|fastmcp)"
```

**Connection Issues:**
```bash
# Test service health
curl http://localhost:8000/health

# Test MCP endpoint
curl http://localhost:8000/mcp/
```

**Claude Desktop Not Connecting:**
1. Verify configuration file location and syntax
2. Restart Claude Desktop after configuration changes  
3. Check that the MCP server is running on port 8000
4. Validate JSON configuration syntax

**Dependency Issues:**
```bash
# Reinstall dependencies
make install

# Check Python version (requires 3.12+)
python --version

# Verify conda environment
conda env list
```

### Debug Commands

```bash
# Run unit tests
make unit-test

# Check application health
curl http://localhost:8000/health

# Enable debug logging
MCP_DEBUG=true make run
```

### Getting Help

1. **Check Documentation**: [docs/](docs/) directory
2. **Review Issues**: GitHub issues for known problems
3. **Enable Debug Logging**: Set `MCP_DEBUG=true`
4. **Use Health Endpoints**: `/health` and `/mcp/` endpoints

## 📚 Documentation

### Core Documentation

| Document | Description |
|----------|-------------|
| **[API Documentation](docs/API.md)** | Complete API reference with examples |
| **[Local Setup Guide](docs/LOCAL.md)** | Local development and deployment guide |
| **[Security Guide](docs/SECURITY.md)** | Security best practices and configurations |
| **[Performance Guide](docs/PERFORMANCE.md)** | Performance optimization and monitoring |
| **[Testing Guide](docs/TESTING.md)** | Testing strategies and best practices |
| **[Development Environment](DEV-ENVIRONMENT.md)** | Local development setup guide |

| **[Changelog](CHANGELOG.md)** | Version history and release notes |

### Quick Links

- **🚀 Getting Started**: [Quick Start](#-quick-start)
- **🚀 Local Deployment**: [Local Setup Guide](docs/LOCAL.md)

- **🔒 Security Setup**: [Security Guide](docs/SECURITY.md)
- **⚡ Performance Tuning**: [Performance Guide](docs/PERFORMANCE.md)
- **🧪 Testing**: [Testing Guide](docs/TESTING.md)

- **📖 API Reference**: [API Documentation](docs/API.md)

### External Resources

- **[FastMCP Documentation](https://github.com/punkpeye/fastmcp)** - MCP framework documentation
- **[XGBoost Documentation](https://xgboost.readthedocs.io/)** - XGBoost library documentation
- **[Claude Desktop](https://claude.ai/download)** - Download Claude Desktop for MCP integration
- **[MCP Specification](https://spec.modelcontextprotocol.io/)** - Model Context Protocol specification

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd mcp-xgboost

# Create development environment
make dev-setup

# Run tests
make test

# Code formatting
make format

# Type checking
make typecheck
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **XGBoost Team**: For the excellent machine learning library
- **FastMCP**: For the modern MCP server framework
- **Python Community**: For the excellent ecosystem and tools

---

**🚀 Ready to get started?** Run `make install && make run` and start training models with Claude Desktop!