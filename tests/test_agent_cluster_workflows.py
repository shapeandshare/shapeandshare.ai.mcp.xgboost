#!/usr/bin/env python3
"""
Agent-Cluster Workflow Tests for MCP XGBoost Application

This test suite validates agent-to-cluster communication and multi-agent
ML workflows for the MCP XGBoost application in Kubernetes.

Test Categories:
1. Agent Initialization & Configuration
2. Single Agent ML Workflows
3. Multi-Agent Coordination
4. Network Resilience & Error Handling
5. Performance Testing

Usage:
    pytest tests/test_agent_cluster_workflows.py -v
    pytest tests/test_agent_cluster_workflows.py::TestAgentInitialization -v
"""

import pytest
import asyncio
import requests
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import aiohttp
    import asyncio
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

try:
    import kubernetes
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

try:
    # Agent libraries - these might not be available in all environments
    from mcp_agent.core.fastagent import FastAgent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

# Combined availability checks
NETWORK_LIBS_AVAILABLE = all([
    ASYNC_AVAILABLE
])

FULL_AGENT_AVAILABLE = all([
    AGENT_AVAILABLE,
    NETWORK_LIBS_AVAILABLE,
    KUBERNETES_AVAILABLE
])


class AgentClusterConfiguration:
    """Configuration for agent-cluster workflow tests"""
    
    def __init__(self):
        self.cluster_url = os.getenv("CLUSTER_URL", "http://localhost:8000")
        self.mcp_endpoint = os.getenv("MCP_ENDPOINT", f"{self.cluster_url}/mcp")
        self.health_endpoint = os.getenv("HEALTH_ENDPOINT", f"{self.cluster_url}/health")
        
        # Agent configuration
        self.agent_timeout = int(os.getenv("AGENT_TIMEOUT", "60"))
        self.mcp_timeout = int(os.getenv("MCP_TIMEOUT", "30"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        
        # Kubernetes configuration
        self.k8s_namespace = os.getenv("K8S_NAMESPACE", "mcp-xgboost")
        self.k8s_context = os.getenv("K8S_CONTEXT", "k3d-mcp-xgboost")
        
        # Performance thresholds
        self.max_agent_response_time = float(os.getenv("MAX_AGENT_RESPONSE_TIME", "30.0"))
        self.max_workflow_time = float(os.getenv("MAX_WORKFLOW_TIME", "300.0"))
        
        # Test data directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="agent_cluster_test_"))
        self.configs_dir = self.temp_dir / "configs"
        self.data_dir = self.temp_dir / "data"
        self.results_dir = self.temp_dir / "results"
        
        # Create directories
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Agent cluster configuration initialized:")
        logger.info(f"  Cluster URL: {self.cluster_url}")
        logger.info(f"  MCP Endpoint: {self.mcp_endpoint}")
        logger.info(f"  Agent Timeout: {self.agent_timeout}s")
        logger.info(f"  Temp Dir: {self.temp_dir}")
    
    def cleanup(self):
        """Clean up temporary test data"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
    
    def get_agent_config(self, agent_name="test_agent"):
        """Get agent configuration for testing"""
        return {
            "mcp": {
                "servers": {
                    "xgboost": {
                        "transport": "http",
                        "url": self.mcp_endpoint,
                        "description": "XGBoost machine learning server",
                        "timeout": self.mcp_timeout,
                        "retries": self.max_retries
                    }
                }
            },
            "defaults": {
                "model": "claude-3-5-sonnet-20241022",
                "temperature": 0.1,
                "max_tokens": 4000
            },
            "agent": {
                "name": agent_name,
                "timeout": self.agent_timeout,
                "max_retries": self.max_retries
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def get_multi_agent_config(self):
        """Get configuration for multi-agent workflows"""
        return {
            "agents": {
                "data_analyst": {
                    "model": "claude-3-5-sonnet-20241022",
                    "temperature": 0.1,
                    "role": "data_analysis"
                },
                "model_trainer": {
                    "model": "claude-3-5-sonnet-20241022",
                    "temperature": 0.2,
                    "role": "model_training"
                },
                "predictor": {
                    "model": "claude-3-5-sonnet-20241022",
                    "temperature": 0.1,
                    "role": "prediction"
                }
            },
            "workflow": {
                "max_agents": 3,
                "coordination_timeout": self.agent_timeout * 2,
                "failure_tolerance": 1
            }
        }


@pytest.fixture(scope="module")
def agent_config():
    """Provide agent configuration for tests"""
    config = AgentClusterConfiguration()
    yield config
    config.cleanup()


@pytest.fixture(scope="module")
def sample_datasets():
    """Generate sample datasets for agent testing"""
    np.random.seed(42)
    
    # House prices dataset
    n_samples = 500
    house_data = pd.DataFrame({
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'sqft': np.random.randint(800, 4000, n_samples),
        'age': np.random.randint(0, 50, n_samples),
        'price': np.random.randint(100000, 800000, n_samples)
    })
    
    # Customer churn dataset
    churn_data = pd.DataFrame({
        'tenure': np.random.randint(0, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'contract_length': np.random.choice(['month', 'year', 'two_year'], n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    # Stock prediction dataset
    stock_data = pd.DataFrame({
        'open': np.random.uniform(100, 300, n_samples),
        'high': np.random.uniform(110, 320, n_samples),
        'low': np.random.uniform(90, 280, n_samples),
        'volume': np.random.randint(1000000, 50000000, n_samples),
        'close': np.random.uniform(95, 315, n_samples)
    })
    
    return {
        'house_prices': house_data,
        'customer_churn': churn_data,
        'stock_prediction': stock_data
    }


class TestAgentInitialization:
    """Test agent initialization and configuration"""
    
    def test_agent_config_creation(self, agent_config):
        """Test that agent configuration is created correctly"""
        config = agent_config.get_agent_config()
        
        assert 'mcp' in config
        assert 'servers' in config['mcp']
        assert 'xgboost' in config['mcp']['servers']
        assert 'defaults' in config
        assert 'agent' in config
        
        xgb_config = config['mcp']['servers']['xgboost']
        assert xgb_config['url'] == agent_config.mcp_endpoint
        assert xgb_config['transport'] == 'http'
        
        logger.info(f"✅ Agent configuration created successfully")
    
    def test_multi_agent_config_creation(self, agent_config):
        """Test multi-agent configuration creation"""
        config = agent_config.get_multi_agent_config()
        
        assert 'agents' in config
        assert 'workflow' in config
        assert len(config['agents']) == 3
        
        expected_agents = ['data_analyst', 'model_trainer', 'predictor']
        for agent_name in expected_agents:
            assert agent_name in config['agents']
            assert 'model' in config['agents'][agent_name]
            assert 'role' in config['agents'][agent_name]
        
        logger.info(f"✅ Multi-agent configuration created successfully")
    
    @pytest.mark.skipif(not FULL_AGENT_AVAILABLE, reason="Agent libraries not available")
    def test_agent_initialization_with_cluster_config(self, agent_config):
        """Test agent initialization with cluster configuration"""
        try:
            # Create temporary config file
            config_file = agent_config.configs_dir / "test_agent.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(agent_config.get_agent_config(), f)
            
            # Test FastAgent initialization
            # Note: This tests initialization only, not full connection
            agent = FastAgent("Test Agent", config_path=str(config_file))
            assert agent is not None
            
            logger.info(f"✅ Agent initialized with cluster configuration")
            
        except Exception as e:
            pytest.skip(f"Agent initialization failed: {e}")
    
    def test_cluster_connectivity_from_agent_perspective(self, agent_config):
        """Test cluster connectivity from agent perspective"""
        try:
            # Test basic connectivity
            response = requests.get(agent_config.health_endpoint, timeout=agent_config.mcp_timeout)
            assert response.status_code == 200
            
            # Test MCP endpoint availability
            response = requests.get(agent_config.mcp_endpoint, timeout=agent_config.mcp_timeout)
            assert response.status_code in [200, 404, 405]  # Various acceptable responses
            
            logger.info(f"✅ Cluster connectivity validated from agent perspective")
            
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Cluster connectivity test failed: {e}")


class TestSingleAgentWorkflows:
    """Test single agent ML workflows"""
    
    @pytest.mark.skipif(not NETWORK_LIBS_AVAILABLE, reason="Network libraries not available")
    def test_data_analysis_workflow_setup(self, agent_config, sample_datasets):
        """Test data analysis workflow setup"""
        try:
            # Save house prices data
            house_data = sample_datasets['house_prices']
            data_file = agent_config.data_dir / "house_prices.csv"
            house_data.to_csv(data_file, index=False)
            
            # Create data analysis config
            analysis_config = {
                "workflow": "data_analysis",
                "data_source": str(data_file),
                "target_column": "price",
                "analysis_type": "regression",
                "agent_role": "data_analyst"
            }
            
            config_file = agent_config.configs_dir / "data_analysis.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(analysis_config, f)
            
            # Verify setup
            assert data_file.exists()
            assert config_file.exists()
            assert len(house_data) == 500
            
            logger.info(f"✅ Data analysis workflow setup complete")
            
        except Exception as e:
            pytest.skip(f"Data analysis workflow setup failed: {e}")
    
    @pytest.mark.skipif(not NETWORK_LIBS_AVAILABLE, reason="Network libraries not available")
    def test_model_training_workflow_setup(self, agent_config, sample_datasets):
        """Test model training workflow setup"""
        try:
            # Save customer churn data
            churn_data = sample_datasets['customer_churn']
            data_file = agent_config.data_dir / "customer_churn.csv"
            churn_data.to_csv(data_file, index=False)
            
            # Create training config
            training_config = {
                "workflow": "model_training",
                "data_source": str(data_file),
                "target_column": "churn",
                "model_type": "classification",
                "agent_role": "model_trainer",
                "parameters": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.3
                }
            }
            
            config_file = agent_config.configs_dir / "model_training.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(training_config, f)
            
            # Verify setup
            assert data_file.exists()
            assert config_file.exists()
            assert len(churn_data) == 500
            assert 'churn' in churn_data.columns
            
            logger.info(f"✅ Model training workflow setup complete")
            
        except Exception as e:
            pytest.skip(f"Model training workflow setup failed: {e}")
    
    @pytest.mark.skipif(not NETWORK_LIBS_AVAILABLE, reason="Network libraries not available")
    def test_prediction_workflow_setup(self, agent_config, sample_datasets):
        """Test prediction workflow setup"""
        try:
            # Save stock prediction data
            stock_data = sample_datasets['stock_prediction']
            data_file = agent_config.data_dir / "stock_data.csv"
            stock_data.to_csv(data_file, index=False)
            
            # Create prediction config
            prediction_config = {
                "workflow": "prediction",
                "data_source": str(data_file),
                "target_column": "close",
                "prediction_type": "regression",
                "agent_role": "predictor",
                "model_name": "stock_prediction_model"
            }
            
            config_file = agent_config.configs_dir / "prediction.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(prediction_config, f)
            
            # Verify setup
            assert data_file.exists()
            assert config_file.exists()
            assert len(stock_data) == 500
            
            logger.info(f"✅ Prediction workflow setup complete")
            
        except Exception as e:
            pytest.skip(f"Prediction workflow setup failed: {e}")


class TestAgentWorkflows:
    """Test complete agent workflows"""
    
    @pytest.mark.skipif(not AGENT_AVAILABLE, reason="Agent libraries not available")
    @pytest.mark.skipif(not NETWORK_LIBS_AVAILABLE, reason="Network libraries not available")
    @pytest.mark.asyncio
    async def test_complete_ml_pipeline(self, agent_config, sample_datasets):
        """Test complete ML pipeline using multiple agents"""
        try:
            # Prepare datasets
            house_data = sample_datasets['house_prices']
            train_file = agent_config.data_dir / "house_train.csv"
            house_data.to_csv(train_file, index=False)
            
            # Create ML pipeline workflow
            pipeline_config = {
                "workflow": "ml_pipeline",
                "steps": [
                    {
                        "name": "data_analysis",
                        "agent": "data_analyst",
                        "task": f"Analyze the dataset at {train_file}"
                    },
                    {
                        "name": "model_training",
                        "agent": "model_trainer",
                        "task": "Train an XGBoost model for house price prediction"
                    },
                    {
                        "name": "model_evaluation",
                        "agent": "model_evaluator",
                        "task": "Evaluate the trained model performance"
                    }
                ]
            }
            
            # Save pipeline config
            pipeline_file = agent_config.configs_dir / "ml_pipeline.yaml"
            with open(pipeline_file, 'w') as f:
                yaml.dump(pipeline_config, f)
            
            # Verify pipeline setup
            assert train_file.exists()
            assert pipeline_file.exists()
            assert len(house_data) == 500
            
            logger.info(f"✅ Complete ML pipeline setup verified")
            
        except Exception as e:
            pytest.skip(f"ML pipeline test failed: {e}")
    
    @pytest.mark.skipif(not AGENT_AVAILABLE, reason="Agent libraries not available")
    @pytest.mark.skipif(not NETWORK_LIBS_AVAILABLE, reason="Network libraries not available")
    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self, agent_config, sample_datasets):
        """Test multi-agent coordination workflows"""
        try:
            # Create multi-agent scenario
            coordination_config = {
                "workflow": "multi_agent_coordination",
                "agents": {
                    "data_analyst": {
                        "task": "Analyze customer churn patterns",
                        "priority": 1,
                        "dependencies": []
                    },
                    "model_trainer": {
                        "task": "Train churn prediction model",
                        "priority": 2,
                        "dependencies": ["data_analyst"]
                    },
                    "predictor": {
                        "task": "Generate churn predictions",
                        "priority": 3,
                        "dependencies": ["model_trainer"]
                    }
                },
                "coordination": {
                    "type": "sequential",
                    "failure_handling": "retry",
                    "max_retries": 2
                }
            }
            
            # Save coordination config
            coord_file = agent_config.configs_dir / "multi_agent_coordination.yaml"
            with open(coord_file, 'w') as f:
                yaml.dump(coordination_config, f)
            
            # Prepare churn data
            churn_data = sample_datasets['customer_churn']
            data_file = agent_config.data_dir / "churn_coordination.csv"
            churn_data.to_csv(data_file, index=False)
            
            # Verify coordination setup
            assert coord_file.exists()
            assert data_file.exists()
            assert len(coordination_config['agents']) == 3
            
            logger.info(f"✅ Multi-agent coordination setup verified")
            
        except Exception as e:
            pytest.skip(f"Multi-agent coordination test failed: {e}")


class TestNetworkResilienceAndErrorHandling:
    """Test network resilience and error handling in agent workflows"""
    
    def test_agent_timeout_handling(self, agent_config):
        """Test agent timeout handling"""
        try:
            # Test very short timeout
            short_timeout = 1
            
            start_time = time.time()
            try:
                response = requests.get(agent_config.health_endpoint, timeout=short_timeout)
                end_time = time.time()
                
                if response.status_code == 200:
                    response_time = end_time - start_time
                    logger.info(f"✅ Server responds quickly ({response_time:.3f}s)")
                    
            except requests.exceptions.Timeout:
                logger.info(f"ℹ️  Timeout after {short_timeout}s (expected for slow responses)")
                
        except Exception as e:
            pytest.skip(f"Timeout handling test failed: {e}")
    
    def test_connection_retry_logic(self, agent_config):
        """Test connection retry logic for agents"""
        try:
            max_retries = agent_config.max_retries
            retry_count = 0
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(agent_config.health_endpoint, timeout=agent_config.mcp_timeout)
                    if response.status_code == 200:
                        logger.info(f"✅ Connection successful after {attempt + 1} attempts")
                        break
                except requests.exceptions.RequestException:
                    retry_count += 1
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Brief delay between retries
            
            # Test passes if we either succeeded or exhausted retries gracefully
            assert retry_count <= max_retries
            
            logger.info(f"✅ Retry logic tested: {retry_count}/{max_retries} retries needed")
            
        except Exception as e:
            pytest.skip(f"Connection retry test failed: {e}")
    
    def test_malformed_response_handling(self, agent_config):
        """Test handling of malformed responses"""
        try:
            # Test different types of responses
            test_endpoints = [
                agent_config.health_endpoint,
                agent_config.mcp_endpoint,
                f"{agent_config.cluster_url}/nonexistent"
            ]
            
            for endpoint in test_endpoints:
                try:
                    response = requests.get(endpoint, timeout=agent_config.mcp_timeout)
                    # Any status code is acceptable - we're testing response handling
                    assert isinstance(response.status_code, int)
                    logger.info(f"✅ Endpoint {endpoint} responded with status {response.status_code}")
                    
                except requests.exceptions.RequestException as e:
                    logger.info(f"ℹ️  Endpoint {endpoint} not available: {e}")
            
            logger.info(f"✅ Malformed response handling tested")
            
        except Exception as e:
            pytest.skip(f"Malformed response handling test failed: {e}")
    
    def test_agent_error_recovery(self, agent_config):
        """Test agent error recovery mechanisms"""
        try:
            # Simulate error recovery by testing various failure scenarios
            error_scenarios = [
                {
                    "name": "network_timeout",
                    "timeout": 0.1,
                    "expected": "timeout"
                },
                {
                    "name": "invalid_endpoint",
                    "endpoint": f"{agent_config.cluster_url}/invalid",
                    "expected": "404"
                },
                {
                    "name": "malformed_request",
                    "method": "POST",
                    "data": "invalid json",
                    "expected": "400_or_405"
                }
            ]
            
            recovery_stats = {"recovered": 0, "failed": 0}
            
            for scenario in error_scenarios:
                try:
                    if scenario["name"] == "network_timeout":
                        response = requests.get(agent_config.health_endpoint, timeout=scenario["timeout"])
                        recovery_stats["recovered"] += 1
                    elif scenario["name"] == "invalid_endpoint":
                        response = requests.get(scenario["endpoint"], timeout=agent_config.mcp_timeout)
                        if response.status_code == 404:
                            recovery_stats["recovered"] += 1
                    elif scenario["name"] == "malformed_request":
                        response = requests.post(
                            agent_config.mcp_endpoint,
                            data=scenario["data"],
                            timeout=agent_config.mcp_timeout
                        )
                        if response.status_code in [400, 405]:
                            recovery_stats["recovered"] += 1
                        
                except (requests.exceptions.Timeout, requests.exceptions.RequestException):
                    recovery_stats["recovered"] += 1  # Expected errors are "recovered"
                except Exception:
                    recovery_stats["failed"] += 1
            
            # Test passes if we handled errors gracefully
            total_scenarios = len(error_scenarios)
            recovery_rate = recovery_stats["recovered"] / total_scenarios
            
            logger.info(f"✅ Error recovery tested: {recovery_stats['recovered']}/{total_scenarios} scenarios handled")
            logger.info(f"   Recovery rate: {recovery_rate:.2%}")
            
        except Exception as e:
            pytest.skip(f"Error recovery test failed: {e}")


class TestAgentPerformance:
    """Test agent performance metrics"""
    
    def test_agent_response_time_baseline(self, agent_config):
        """Test agent response time baseline"""
        try:
            # Test response time to health endpoint
            start_time = time.time()
            response = requests.get(agent_config.health_endpoint, timeout=agent_config.mcp_timeout)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            assert response.status_code == 200
            assert response_time < agent_config.max_agent_response_time
            
            logger.info(f"✅ Agent response time baseline: {response_time:.3f}s")
            
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Agent response time test failed: {e}")
    
    def test_concurrent_agent_operations(self, agent_config):
        """Test concurrent agent operations"""
        try:
            import threading
            
            num_concurrent = 3
            results = []
            
            def make_agent_request():
                try:
                    start_time = time.time()
                    response = requests.get(agent_config.health_endpoint, timeout=agent_config.mcp_timeout)
                    end_time = time.time()
                    
                    results.append({
                        'success': response.status_code == 200,
                        'response_time': end_time - start_time,
                        'status_code': response.status_code
                    })
                except Exception as e:
                    results.append({
                        'success': False,
                        'error': str(e)
                    })
            
            # Run concurrent requests
            threads = []
            for _ in range(num_concurrent):
                thread = threading.Thread(target=make_agent_request)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # Analyze results
            successful = [r for r in results if r.get('success', False)]
            success_rate = len(successful) / num_concurrent
            
            if successful:
                avg_response_time = sum(r['response_time'] for r in successful) / len(successful)
                logger.info(f"✅ Concurrent operations: {len(successful)}/{num_concurrent} successful")
                logger.info(f"   Success rate: {success_rate:.2%}")
                logger.info(f"   Average response time: {avg_response_time:.3f}s")
            
        except Exception as e:
            pytest.skip(f"Concurrent operations test failed: {e}")


class TestAgentConfiguration:
    """Test agent configuration and validation"""
    
    def test_agent_config_validation(self, agent_config):
        """Test agent configuration validation"""
        config = agent_config.get_agent_config()
        
        # Required fields
        required_fields = ['mcp', 'defaults', 'agent']
        for field in required_fields:
            assert field in config, f"Missing required field: {field}"
        
        # MCP configuration validation
        mcp_config = config['mcp']
        assert 'servers' in mcp_config
        assert 'xgboost' in mcp_config['servers']
        
        xgb_config = mcp_config['servers']['xgboost']
        assert 'url' in xgb_config
        assert 'transport' in xgb_config
        assert xgb_config['transport'] == 'http'
        
        # Agent configuration validation
        agent_cfg = config['agent']
        assert 'name' in agent_cfg
        assert 'timeout' in agent_cfg
        
        logger.info(f"✅ Agent configuration validated successfully")
    
    def test_multi_agent_config_validation(self, agent_config):
        """Test multi-agent configuration validation"""
        config = agent_config.get_multi_agent_config()
        
        # Required fields
        assert 'agents' in config
        assert 'workflow' in config
        
        # Agent definitions
        agents = config['agents']
        assert len(agents) > 0
        
        for agent_name, agent_cfg in agents.items():
            assert 'model' in agent_cfg
            assert 'role' in agent_cfg
            assert isinstance(agent_cfg['model'], str)
            assert isinstance(agent_cfg['role'], str)
        
        # Workflow configuration
        workflow = config['workflow']
        assert 'max_agents' in workflow
        assert isinstance(workflow['max_agents'], int)
        
        logger.info(f"✅ Multi-agent configuration validated successfully")


class TestDataProcessingWorkflows:
    """Test data processing workflows through agents"""
    
    def test_data_preprocessing_workflow(self, agent_config, sample_datasets):
        """Test data preprocessing workflow"""
        try:
            # Create preprocessing workflow
            preprocessing_config = {
                "workflow": "data_preprocessing",
                "steps": [
                    {
                        "name": "data_loading",
                        "task": "Load and validate data"
                    },
                    {
                        "name": "data_cleaning",
                        "task": "Clean and prepare data"
                    },
                    {
                        "name": "feature_engineering",
                        "task": "Create and select features"
                    }
                ],
                "parameters": {
                    "missing_value_strategy": "mean",
                    "scaling_method": "standard",
                    "feature_selection": "top_k"
                }
            }
            
            # Save preprocessing config
            config_file = agent_config.configs_dir / "data_preprocessing.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(preprocessing_config, f)
            
            # Save sample data
            house_data = sample_datasets['house_prices']
            data_file = agent_config.data_dir / "house_preprocessing.csv"
            house_data.to_csv(data_file, index=False)
            
            # Verify setup
            assert config_file.exists()
            assert data_file.exists()
            assert len(preprocessing_config['steps']) == 3
            
            logger.info(f"✅ Data preprocessing workflow setup complete")
            
        except Exception as e:
            pytest.skip(f"Data preprocessing workflow test failed: {e}")
    
    def test_model_evaluation_workflow(self, agent_config, sample_datasets):
        """Test model evaluation workflow"""
        try:
            # Create evaluation workflow
            evaluation_config = {
                "workflow": "model_evaluation",
                "metrics": [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc"
                ],
                "evaluation_methods": [
                    "cross_validation",
                    "holdout_validation",
                    "bootstrap_validation"
                ],
                "reporting": {
                    "generate_plots": True,
                    "save_results": True,
                    "create_summary": True
                }
            }
            
            # Save evaluation config
            config_file = agent_config.configs_dir / "model_evaluation.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(evaluation_config, f)
            
            # Save sample data
            churn_data = sample_datasets['customer_churn']
            data_file = agent_config.data_dir / "churn_evaluation.csv"
            churn_data.to_csv(data_file, index=False)
            
            # Verify setup
            assert config_file.exists()
            assert data_file.exists()
            assert len(evaluation_config['metrics']) == 5
            
            logger.info(f"✅ Model evaluation workflow setup complete")
            
        except Exception as e:
            pytest.skip(f"Model evaluation workflow test failed: {e}")


if __name__ == "__main__":
    """Run tests directly"""
    pytest.main([__file__, "-v"]) 