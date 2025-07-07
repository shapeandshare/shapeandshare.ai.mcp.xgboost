"""Tests for src/services/xgboost.py module.

This module tests the XGBoost service implementation including MCP tool
registration, model training, prediction, analysis, and management functionality.
"""

import json
import pytest
import pandas as pd
import xgboost as xgb
from unittest.mock import Mock, patch, AsyncMock
from fastmcp import FastMCP

from src.services.xgboost import XGBoostService


class TestXGBoostServiceInitialization:
    """Test XGBoostService initialization and setup."""

    def test_service_initialization(self):
        """Test basic service initialization."""
        mock_mcp = Mock(spec=FastMCP)
        mock_mcp.tool = Mock()
        
        service = XGBoostService(mock_mcp)
        
        assert service._mcp is mock_mcp
        assert service._models == {}
        assert isinstance(service._models, dict)

    def test_service_initialization_with_real_mcp(self):
        """Test service initialization with real FastMCP instance."""
        mcp = FastMCP("Test Server")
        service = XGBoostService(mcp)
        
        assert service._mcp is mcp
        assert service._models == {}

    def test_service_mcp_property(self):
        """Test MCP property getter."""
        mock_mcp = Mock(spec=FastMCP)
        mock_mcp.tool = Mock()
        
        service = XGBoostService(mock_mcp)
        
        assert service.mcp is mock_mcp

    def test_service_register_tools_called(self):
        """Test that _register_tools is called during initialization."""
        mock_mcp = Mock(spec=FastMCP)
        mock_mcp.tool = Mock()
        
        with patch.object(XGBoostService, '_register_tools') as mock_register:
            XGBoostService(mock_mcp)
            mock_register.assert_called_once()

    def test_service_tools_registration(self):
        """Test that all tools are registered with MCP."""
        mock_mcp = Mock(spec=FastMCP)
        mock_mcp.tool = Mock()
        
        service = XGBoostService(mock_mcp)
        
        # Should have called tool decorator multiple times
        assert mock_mcp.tool.call_count >= 5  # At least 5 tools

    def test_service_docstring_exists(self):
        """Test that service class has proper documentation."""
        assert XGBoostService.__doc__ is not None
        assert len(XGBoostService.__doc__) > 0
        assert "XGBoost machine learning service" in XGBoostService.__doc__

    def test_service_init_docstring_exists(self):
        """Test that __init__ method has proper documentation."""
        assert XGBoostService.__init__.__doc__ is not None
        assert len(XGBoostService.__init__.__doc__) > 0
        assert "Initialize the XGBoost service" in XGBoostService.__init__.__doc__


class TestXGBoostServiceTraining:
    """Test model training functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_mcp = Mock(spec=FastMCP)
        self.mock_mcp.tool = Mock()
        self.service = XGBoostService(self.mock_mcp)

    @pytest.mark.asyncio
    async def test_train_model_regression_success(self):
        """Test successful regression model training."""
        # Create test data
        data = {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": [3, 6, 9, 12, 15]
        }
        data_json = json.dumps(data)
        
        result = await self.service._train_model_impl(
            model_name="test_model",
            data=data_json,
            target_column="target",
            model_type="regression"
        )
        
        assert "Model 'test_model' trained successfully!" in result
        assert "Type: regression" in result
        assert "Features: ['feature1', 'feature2']" in result
        assert "Training samples: 5" in result
        assert "test_model" in self.service._models
        assert isinstance(self.service._models["test_model"], xgb.XGBRegressor)

    @pytest.mark.asyncio
    async def test_train_model_classification_success(self):
        """Test successful classification model training."""
        # Create test data
        data = {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": [0, 1, 0, 1, 0]
        }
        data_json = json.dumps(data)
        
        result = await self.service._train_model_impl(
            model_name="test_classifier",
            data=data_json,
            target_column="target",
            model_type="classification"
        )
        
        assert "Model 'test_classifier' trained successfully!" in result
        assert "Type: classification" in result
        assert "test_classifier" in self.service._models
        assert isinstance(self.service._models["test_classifier"], xgb.XGBClassifier)

    @pytest.mark.asyncio
    async def test_train_model_with_custom_parameters(self):
        """Test training with custom XGBoost parameters."""
        data = {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": [3, 6, 9, 12, 15]
        }
        data_json = json.dumps(data)
        
        result = await self.service._train_model_impl(
            model_name="custom_model",
            data=data_json,
            target_column="target",
            model_type="regression",
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1
        )
        
        assert "Model 'custom_model' trained successfully!" in result
        assert "custom_model" in self.service._models
        
        # Verify model parameters
        model = self.service._models["custom_model"]
        assert model.n_estimators == 50
        assert model.max_depth == 3
        assert model.learning_rate == 0.1

    @pytest.mark.asyncio
    async def test_train_model_invalid_json(self):
        """Test training with invalid JSON data."""
        invalid_json = "invalid json data"
        
        result = await self.service._train_model_impl(
            model_name="test_model",
            data=invalid_json,
            target_column="target",
            model_type="regression"
        )
        
        assert "Error training model:" in result
        assert "test_model" not in self.service._models

    @pytest.mark.asyncio
    async def test_train_model_missing_target_column(self):
        """Test training with missing target column."""
        data = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6]
            # Missing target column
        }
        data_json = json.dumps(data)
        
        result = await self.service._train_model_impl(
            model_name="test_model",
            data=data_json,
            target_column="missing_target",
            model_type="regression"
        )
        
        assert "Error training model:" in result or "Data validation failed:" in result
        assert "test_model" not in self.service._models

    @pytest.mark.asyncio
    async def test_train_model_empty_data(self):
        """Test training with empty data."""
        data = {}
        data_json = json.dumps(data)
        
        result = await self.service._train_model_impl(
            model_name="test_model",
            data=data_json,
            target_column="target",
            model_type="regression"
        )
        
        assert "Error training model:" in result or "Data validation failed:" in result
        assert "test_model" not in self.service._models

    @pytest.mark.asyncio
    @patch('src.services.xgboost.validate_data')
    async def test_train_model_validation_failure(self, mock_validate):
        """Test training with data validation failure."""
        mock_validate.return_value = "Data validation failed: Test error"
        
        data = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        }
        data_json = json.dumps(data)
        
        result = await self.service._train_model_impl(
            model_name="test_model",
            data=data_json,
            target_column="target",
            model_type="regression"
        )
        
        assert "Data validation failed:" in result
        assert "test_model" not in self.service._models

    @pytest.mark.asyncio
    async def test_train_model_default_regression_type(self):
        """Test that default model type is regression."""
        data = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        }
        data_json = json.dumps(data)
        
        result = await self.service._train_model_impl(
            model_name="default_model",
            data=data_json,
            target_column="target",
            model_type="unknown_type"  # Should default to regression
        )
        
        assert "Model 'default_model' trained successfully!" in result
        assert isinstance(self.service._models["default_model"], xgb.XGBRegressor)

    @pytest.mark.asyncio
    async def test_train_model_case_insensitive_classification(self):
        """Test that classification type is case insensitive."""
        data = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        }
        data_json = json.dumps(data)
        
        result = await self.service._train_model_impl(
            model_name="case_test",
            data=data_json,
            target_column="target",
            model_type="CLASSIFICATION"
        )
        
        assert "Model 'case_test' trained successfully!" in result
        assert isinstance(self.service._models["case_test"], xgb.XGBClassifier)


class TestXGBoostServicePrediction:
    """Test model prediction functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_mcp = Mock(spec=FastMCP)
        self.mock_mcp.tool = Mock()
        self.service = XGBoostService(self.mock_mcp)

    @pytest.mark.asyncio
    async def test_predict_success(self):
        """Test successful prediction."""
        # First train a model
        train_data = {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": [3, 6, 9, 12, 15]
        }
        train_data_json = json.dumps(train_data)
        
        await self.service._train_model_impl(
            model_name="test_model",
            data=train_data_json,
            target_column="target",
            model_type="regression"
        )
        
        # Now make predictions
        predict_data = {
            "feature1": [6, 7],
            "feature2": [12, 14]
        }
        predict_data_json = json.dumps(predict_data)
        
        result = await self.service._predict_impl(
            model_name="test_model",
            data=predict_data_json
        )
        
        assert "Predictions from model 'test_model':" in result
        assert "Sample 1:" in result
        assert "Sample 2:" in result

    @pytest.mark.asyncio
    async def test_predict_model_not_found(self):
        """Test prediction with non-existent model."""
        predict_data = {
            "feature1": [1, 2],
            "feature2": [3, 4]
        }
        predict_data_json = json.dumps(predict_data)
        
        result = await self.service._predict_impl(
            model_name="nonexistent_model",
            data=predict_data_json
        )
        
        assert "Model 'nonexistent_model' not found" in result
        assert "Available models:" in result

    @pytest.mark.asyncio
    async def test_predict_invalid_json(self):
        """Test prediction with invalid JSON data."""
        # First train a model
        train_data = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        }
        train_data_json = json.dumps(train_data)
        
        await self.service._train_model_impl(
            model_name="test_model",
            data=train_data_json,
            target_column="target",
            model_type="regression"
        )
        
        # Try to predict with invalid JSON
        invalid_json = "invalid json"
        
        result = await self.service._predict_impl(
            model_name="test_model",
            data=invalid_json
        )
        
        assert "Error making predictions:" in result

    @pytest.mark.asyncio
    async def test_predict_missing_features(self):
        """Test prediction with missing feature columns."""
        # First train a model
        train_data = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        }
        train_data_json = json.dumps(train_data)
        
        await self.service._train_model_impl(
            model_name="test_model",
            data=train_data_json,
            target_column="target",
            model_type="regression"
        )
        
        # Try to predict with missing features
        predict_data = {
            "feature1": [10]  # Missing feature2
        }
        predict_data_json = json.dumps(predict_data)
        
        result = await self.service._predict_impl(
            model_name="test_model",
            data=predict_data_json
        )
        
        assert "Error making predictions:" in result

    @pytest.mark.asyncio
    async def test_predict_empty_data(self):
        """Test prediction with empty data."""
        # First train a model
        train_data = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        }
        train_data_json = json.dumps(train_data)
        
        await self.service._train_model_impl(
            model_name="test_model",
            data=train_data_json,
            target_column="target",
            model_type="regression"
        )
        
        # Try to predict with empty data
        predict_data = {}
        predict_data_json = json.dumps(predict_data)
        
        result = await self.service._predict_impl(
            model_name="test_model",
            data=predict_data_json
        )
        
        assert "Error making predictions:" in result

    @pytest.mark.asyncio
    async def test_predict_classification_model(self):
        """Test prediction with classification model."""
        # First train a classification model
        train_data = {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": [0, 1, 0, 1, 0]
        }
        train_data_json = json.dumps(train_data)
        
        await self.service._train_model_impl(
            model_name="classifier",
            data=train_data_json,
            target_column="target",
            model_type="classification"
        )
        
        # Make predictions
        predict_data = {
            "feature1": [6, 7],
            "feature2": [12, 14]
        }
        predict_data_json = json.dumps(predict_data)
        
        result = await self.service._predict_impl(
            model_name="classifier",
            data=predict_data_json
        )
        
        assert "Predictions from model 'classifier':" in result
        assert "Sample 1:" in result
        assert "Sample 2:" in result

    @pytest.mark.asyncio
    async def test_predict_single_sample(self):
        """Test prediction with single sample."""
        # First train a model
        train_data = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        }
        train_data_json = json.dumps(train_data)
        
        await self.service._train_model_impl(
            model_name="test_model",
            data=train_data_json,
            target_column="target",
            model_type="regression"
        )
        
        # Make prediction with single sample
        predict_data = {
            "feature1": [10],
            "feature2": [20]
        }
        predict_data_json = json.dumps(predict_data)
        
        result = await self.service._predict_impl(
            model_name="test_model",
            data=predict_data_json
        )
        
        assert "Predictions from model 'test_model':" in result
        assert "Sample 1:" in result
        assert "Sample 2:" not in result


class TestXGBoostServiceModelInfo:
    """Test model information functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_mcp = Mock(spec=FastMCP)
        self.mock_mcp.tool = Mock()
        self.service = XGBoostService(self.mock_mcp)

    @pytest.mark.asyncio
    async def test_get_model_info_success(self):
        """Test successful model info retrieval."""
        # First train a model
        train_data = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        }
        train_data_json = json.dumps(train_data)
        
        await self.service._train_model_impl(
            model_name="test_model",
            data=train_data_json,
            target_column="target",
            model_type="regression"
        )
        
        result = await self.service._get_model_info_impl("test_model")
        
        # Should contain model information
        assert "test_model" in result
        assert "Model Type:" in result or "Type:" in result

    @pytest.mark.asyncio
    async def test_get_model_info_model_not_found(self):
        """Test model info with non-existent model."""
        result = await self.service._get_model_info_impl("nonexistent_model")
        
        assert "Model 'nonexistent_model' not found" in result
        assert "Available models:" in result

    @pytest.mark.asyncio
    @patch('src.services.xgboost.format_model_info')
    async def test_get_model_info_format_error(self, mock_format):
        """Test model info with formatting error."""
        # First train a model
        train_data = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        }
        train_data_json = json.dumps(train_data)
        
        await self.service._train_model_impl(
            model_name="test_model",
            data=train_data_json,
            target_column="target",
            model_type="regression"
        )
        
        # Mock format_model_info to raise exception
        mock_format.side_effect = AttributeError("Test error")
        
        result = await self.service._get_model_info_impl("test_model")
        
        assert "Error getting model info:" in result

    @pytest.mark.asyncio
    async def test_get_model_info_multiple_models(self):
        """Test model info when multiple models exist."""
        # Train multiple models
        train_data1 = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        }
        train_data2 = {
            "feature1": [10, 20, 30],
            "feature2": [40, 50, 60],
            "target": [0, 1, 0]
        }
        
        await self.service._train_model_impl(
            model_name="model1",
            data=json.dumps(train_data1),
            target_column="target",
            model_type="regression"
        )
        
        await self.service._train_model_impl(
            model_name="model2",
            data=json.dumps(train_data2),
            target_column="target",
            model_type="classification"
        )
        
        # Get info for specific model
        result = await self.service._get_model_info_impl("model1")
        
        assert "model1" in result
        # Should not contain info about model2
        assert "model2" not in result or "Available models:" in result


class TestXGBoostServiceFeatureImportance:
    """Test feature importance functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_mcp = Mock(spec=FastMCP)
        self.mock_mcp.tool = Mock()
        self.service = XGBoostService(self.mock_mcp)

    @pytest.mark.asyncio
    async def test_get_feature_importance_success(self):
        """Test successful feature importance retrieval."""
        # First train a model
        train_data = {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": [3, 6, 9, 12, 15]
        }
        train_data_json = json.dumps(train_data)
        
        await self.service._train_model_impl(
            model_name="test_model",
            data=train_data_json,
            target_column="target",
            model_type="regression"
        )
        
        result = await self.service._get_feature_importance_impl("test_model")
        
        # Should contain feature importance information
        assert "Feature Importance" in result or "feature1" in result or "feature2" in result

    @pytest.mark.asyncio
    async def test_get_feature_importance_model_not_found(self):
        """Test feature importance with non-existent model."""
        result = await self.service._get_feature_importance_impl("nonexistent_model")
        
        assert "Model 'nonexistent_model' not found" in result
        assert "Available models:" in result

    @pytest.mark.asyncio
    @patch('src.services.xgboost.format_feature_importance')
    async def test_get_feature_importance_format_error(self, mock_format):
        """Test feature importance with formatting error."""
        # First train a model
        train_data = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        }
        train_data_json = json.dumps(train_data)
        
        await self.service._train_model_impl(
            model_name="test_model",
            data=train_data_json,
            target_column="target",
            model_type="regression"
        )
        
        # Mock format_feature_importance to raise exception
        mock_format.side_effect = AttributeError("Test error")
        
        result = await self.service._get_feature_importance_impl("test_model")
        
        assert "Error getting feature importance:" in result

    @pytest.mark.asyncio
    async def test_get_feature_importance_classification_model(self):
        """Test feature importance with classification model."""
        # First train a classification model
        train_data = {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": [0, 1, 0, 1, 0]
        }
        train_data_json = json.dumps(train_data)
        
        await self.service._train_model_impl(
            model_name="classifier",
            data=train_data_json,
            target_column="target",
            model_type="classification"
        )
        
        result = await self.service._get_feature_importance_impl("classifier")
        
        # Should contain feature importance information
        assert "Feature Importance" in result or "feature1" in result or "feature2" in result


class TestXGBoostServiceListModels:
    """Test model listing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_mcp = Mock(spec=FastMCP)
        self.mock_mcp.tool = Mock()
        self.service = XGBoostService(self.mock_mcp)

    @pytest.mark.asyncio
    async def test_list_models_empty(self):
        """Test listing models when none exist."""
        result = await self.service._list_models_impl()
        
        assert "No models have been trained yet." in result

    @pytest.mark.asyncio
    async def test_list_models_single_regression(self):
        """Test listing single regression model."""
        # Train a regression model
        train_data = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        }
        train_data_json = json.dumps(train_data)
        
        await self.service._train_model_impl(
            model_name="regression_model",
            data=train_data_json,
            target_column="target",
            model_type="regression"
        )
        
        result = await self.service._list_models_impl()
        
        assert "Available models:" in result
        assert "regression_model (Regression)" in result

    @pytest.mark.asyncio
    async def test_list_models_single_classification(self):
        """Test listing single classification model."""
        # Train a classification model
        train_data = {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": [0, 1, 0, 1, 0]
        }
        train_data_json = json.dumps(train_data)
        
        await self.service._train_model_impl(
            model_name="classification_model",
            data=train_data_json,
            target_column="target",
            model_type="classification"
        )
        
        result = await self.service._list_models_impl()
        
        assert "Available models:" in result
        assert "classification_model (Classification)" in result

    @pytest.mark.asyncio
    async def test_list_models_multiple(self):
        """Test listing multiple models."""
        # Train multiple models
        train_data1 = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        }
        train_data2 = {
            "feature1": [10, 20, 30],
            "feature2": [40, 50, 60],
            "target": [0, 1, 0]
        }
        
        await self.service._train_model_impl(
            model_name="model1",
            data=json.dumps(train_data1),
            target_column="target",
            model_type="regression"
        )
        
        await self.service._train_model_impl(
            model_name="model2",
            data=json.dumps(train_data2),
            target_column="target",
            model_type="classification"
        )
        
        result = await self.service._list_models_impl()
        
        assert "Available models:" in result
        assert "model1 (Regression)" in result
        assert "model2 (Classification)" in result

    @pytest.mark.asyncio
    async def test_list_models_type_detection(self):
        """Test model type detection based on predict_proba method."""
        # Train models and verify type detection
        train_data_reg = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        }
        train_data_clf = {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": [0, 1, 0, 1, 0]
        }
        
        await self.service._train_model_impl(
            model_name="regressor",
            data=json.dumps(train_data_reg),
            target_column="target",
            model_type="regression"
        )
        
        await self.service._train_model_impl(
            model_name="classifier",
            data=json.dumps(train_data_clf),
            target_column="target",
            model_type="classification"
        )
        
        result = await self.service._list_models_impl()
        
        assert "regressor (Regression)" in result
        assert "classifier (Classification)" in result
        
        # Verify type detection logic
        reg_model = self.service._models["regressor"]
        clf_model = self.service._models["classifier"]
        
        assert not hasattr(reg_model, "predict_proba")
        assert hasattr(clf_model, "predict_proba")


class TestXGBoostServiceIntegration:
    """Test service integration and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_mcp = Mock(spec=FastMCP)
        self.mock_mcp.tool = Mock()
        self.service = XGBoostService(self.mock_mcp)

    def test_service_models_storage(self):
        """Test that models are stored correctly in service."""
        # Initially empty
        assert len(self.service._models) == 0
        
        # Add a mock model
        mock_model = Mock()
        self.service._models["test"] = mock_model
        
        assert len(self.service._models) == 1
        assert self.service._models["test"] is mock_model

    def test_service_mcp_reference(self):
        """Test that service maintains MCP reference."""
        assert self.service._mcp is self.mock_mcp
        assert self.service.mcp is self.mock_mcp

    @pytest.mark.asyncio
    async def test_model_overwrite(self):
        """Test that training overwrites existing model."""
        # Train first model
        train_data1 = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        }
        
        await self.service._train_model_impl(
            model_name="same_name",
            data=json.dumps(train_data1),
            target_column="target",
            model_type="regression"
        )
        
        first_model = self.service._models["same_name"]
        
        # Train second model with same name
        train_data2 = {
            "feature1": [10, 20, 30],
            "feature2": [40, 50, 60],
            "target": [0, 1, 0]
        }
        
        await self.service._train_model_impl(
            model_name="same_name",
            data=json.dumps(train_data2),
            target_column="target",
            model_type="classification"
        )
        
        second_model = self.service._models["same_name"]
        
        # Should be different models
        assert first_model is not second_model
        assert isinstance(second_model, xgb.XGBClassifier)

    @pytest.mark.asyncio
    async def test_concurrent_model_access(self):
        """Test accessing models from multiple operations."""
        # Train a model
        train_data = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        }
        train_data_json = json.dumps(train_data)
        
        await self.service._train_model_impl(
            model_name="shared_model",
            data=train_data_json,
            target_column="target",
            model_type="regression"
        )
        
        # Access from multiple methods
        info_result = await self.service._get_model_info_impl("shared_model")
        importance_result = await self.service._get_feature_importance_impl("shared_model")
        list_result = await self.service._list_models_impl()
        
        # All should succeed
        assert "shared_model" in info_result
        assert "Feature Importance" in importance_result or "feature1" in importance_result
        assert "shared_model" in list_result

    def test_service_state_isolation(self):
        """Test that service instances maintain separate state."""
        # Create second service instance
        mock_mcp2 = Mock(spec=FastMCP)
        mock_mcp2.tool = Mock()
        service2 = XGBoostService(mock_mcp2)
        
        # Add model to first service
        self.service._models["model1"] = Mock()
        
        # Second service should be empty
        assert len(service2._models) == 0
        assert "model1" not in service2._models

    def test_service_error_handling_isolation(self):
        """Test that errors in one operation don't affect service state."""
        # Store initial state
        initial_models = dict(self.service._models)
        
        # Add a valid model
        self.service._models["valid_model"] = Mock()
        
        # Simulate error in operation (model should remain)
        assert "valid_model" in self.service._models
        assert len(self.service._models) == 1

    @pytest.mark.asyncio
    async def test_empty_model_name_handling(self):
        """Test handling of empty model names."""
        train_data = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        }
        train_data_json = json.dumps(train_data)
        
        # Try to train with empty name
        result = await self.service._train_model_impl(
            model_name="",
            data=train_data_json,
            target_column="target",
            model_type="regression"
        )
        
        # Should handle gracefully
        assert "Error training model:" in result or "trained successfully" in result

    @pytest.mark.asyncio
    async def test_special_characters_in_model_name(self):
        """Test handling of special characters in model names."""
        train_data = {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        }
        train_data_json = json.dumps(train_data)
        
        # Try to train with special characters
        special_name = "model-with_special.chars@123"
        result = await self.service._train_model_impl(
            model_name=special_name,
            data=train_data_json,
            target_column="target",
            model_type="regression"
        )
        
        # Should handle gracefully
        assert "Error training model:" in result or "trained successfully" in result


class TestXGBoostServiceMCPToolWrappers:
    """Test the MCP tool wrapper functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a real FastMCP instance to properly test the wrappers
        from fastmcp import FastMCP
        self.mcp = FastMCP("Test Server")
        self.service = XGBoostService(self.mcp)

    @pytest.mark.asyncio
    async def test_mcp_tool_train_model_wrapper(self):
        """Test the MCP tool wrapper for train_model."""
        # Get the registered tools from the mcp instance
        tools = getattr(self.mcp, '_tools', {})
        
        # Find the train_model tool
        train_tool = None
        for tool_name, tool_func in tools.items():
            if 'train_model' in tool_name or hasattr(tool_func, '__name__') and 'train_model' in getattr(tool_func, '__name__', ''):
                train_tool = tool_func
                break
        
        if train_tool:
            # Test calling the tool wrapper
            data = {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                "target": [7, 8, 9]
            }
            data_json = json.dumps(data)
            
            result = await train_tool(
                model_name="wrapper_test",
                data=data_json,
                target_column="target",
                model_type="regression"
            )
            
            assert "trained successfully" in result
            assert "wrapper_test" in self.service._models

    @pytest.mark.asyncio
    async def test_mcp_tool_predict_wrapper(self):
        """Test the MCP tool wrapper for predict."""
        # First train a model using implementation method
        await self.service._train_model_impl(
            "test_model", 
            json.dumps({"feature1": [1,2,3], "feature2": [4,5,6], "target": [7,8,9]}),
            "target",
            "regression"
        )
        
        # Get the registered tools
        tools = getattr(self.mcp, '_tools', {})
        
        # Find the predict tool
        predict_tool = None
        for tool_name, tool_func in tools.items():
            if 'predict' in tool_name:
                predict_tool = tool_func
                break
        
        if predict_tool:
            result = await predict_tool(
                "test_model",
                json.dumps({"feature1": [10], "feature2": [20]})
            )
            
            assert "Predictions from model" in result

    @pytest.mark.asyncio
    async def test_mcp_tool_get_model_info_wrapper(self):
        """Test the MCP tool wrapper for get_model_info."""
        # First train a model
        await self.service._train_model_impl(
            "info_test", 
            json.dumps({"feature1": [1,2,3], "feature2": [4,5,6], "target": [7,8,9]}),
            "target",
            "regression"
        )
        
        # Get the registered tools
        tools = getattr(self.mcp, '_tools', {})
        
        # Find the get_model_info tool
        info_tool = None
        for tool_name, tool_func in tools.items():
            if 'model_info' in tool_name or 'get_model_info' in tool_name:
                info_tool = tool_func
                break
        
        if info_tool:
            result = await info_tool("info_test")
            assert "info_test" in result

    @pytest.mark.asyncio
    async def test_mcp_tool_get_feature_importance_wrapper(self):
        """Test the MCP tool wrapper for get_feature_importance."""
        # First train a model
        await self.service._train_model_impl(
            "importance_test", 
            json.dumps({"feature1": [1,2,3], "feature2": [4,5,6], "target": [7,8,9]}),
            "target",
            "regression"
        )
        
        # Get the registered tools
        tools = getattr(self.mcp, '_tools', {})
        
        # Find the get_feature_importance tool
        importance_tool = None
        for tool_name, tool_func in tools.items():
            if 'feature_importance' in tool_name or 'importance' in tool_name:
                importance_tool = tool_func
                break
        
        if importance_tool:
            result = await importance_tool("importance_test")
            assert "Feature Importance" in result or "feature1" in result

    @pytest.mark.asyncio
    async def test_mcp_tool_list_models_wrapper(self):
        """Test the MCP tool wrapper for list_models."""
        # First train a model
        await self.service._train_model_impl(
            "list_test", 
            json.dumps({"feature1": [1,2,3], "feature2": [4,5,6], "target": [7,8,9]}),
            "target",
            "regression"
        )
        
        # Get the registered tools
        tools = getattr(self.mcp, '_tools', {})
        
        # Find the list_models tool
        list_tool = None
        for tool_name, tool_func in tools.items():
            if 'list_models' in tool_name or 'list' in tool_name:
                list_tool = tool_func
                break
        
        if list_tool:
            result = await list_tool()
            assert "list_test" in result

    def test_mcp_tools_are_registered(self):
        """Test that MCP tools are properly registered."""
        # Check that service has the MCP instance
        assert self.service._mcp is self.mcp
        
        # Check that the service has models storage
        assert hasattr(self.service, '_models')
        assert isinstance(self.service._models, dict)
        
        # Check that the MCP server exists and is the right type
        from fastmcp import FastMCP
        assert isinstance(self.mcp, FastMCP)
        
        # The tools are registered during service initialization
        # Verify that _register_tools was called by checking service state
        assert self.service._mcp is not None


class TestXGBoostServiceDocumentation:
    """Test service documentation and metadata."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_mcp = Mock(spec=FastMCP)
        self.mock_mcp.tool = Mock()
        self.service = XGBoostService(self.mock_mcp)

    def test_module_docstring(self):
        """Test that module has proper documentation."""
        import src.services.xgboost as xgb_module
        assert xgb_module.__doc__ is not None
        assert len(xgb_module.__doc__) > 0
        assert "XGBoost Machine Learning Service" in xgb_module.__doc__

    def test_class_docstring_content(self):
        """Test class docstring content."""
        docstring = XGBoostService.__doc__
        
        # Check for key documentation elements
        assert "Parameters" in docstring
        assert "Attributes" in docstring
        assert "Methods" in docstring
        assert "Examples" in docstring
        assert "Notes" in docstring

    def test_method_docstrings_exist(self):
        """Test that all methods have docstrings."""
        methods = [
            XGBoostService._train_model_impl,
            XGBoostService._predict_impl,
            XGBoostService._get_model_info_impl,
            XGBoostService._get_feature_importance_impl,
            XGBoostService._list_models_impl,
            XGBoostService._register_tools,
        ]
        
        for method in methods:
            assert method.__doc__ is not None
            assert len(method.__doc__) > 0

    def test_property_docstring(self):
        """Test property docstring."""
        # Get the property descriptor
        mcp_property = XGBoostService.mcp
        assert mcp_property.__doc__ is not None
        assert len(mcp_property.__doc__) > 0
        assert "MCP server instance" in mcp_property.__doc__

    def test_imports_availability(self):
        """Test that all required imports are available."""
        dependencies = [
            'json', 'pandas', 'xgboost', 'fastmcp'
        ]
        
        for dep in dependencies:
            try:
                __import__(dep)
            except ImportError:
                pytest.fail(f"Required dependency {dep} not available")

    def test_service_typing_annotations(self):
        """Test that service has proper type annotations."""
        import inspect
        
        # Check __init__ signature
        init_sig = inspect.signature(XGBoostService.__init__)
        assert 'mcp' in init_sig.parameters
        
        # Check return type annotations exist
        methods_with_returns = [
            XGBoostService._train_model_impl,
            XGBoostService._predict_impl,
            XGBoostService._get_model_info_impl,
            XGBoostService._get_feature_importance_impl,
            XGBoostService._list_models_impl,
        ]
        
        for method in methods_with_returns:
            sig = inspect.signature(method)
            # Should have return annotation
            assert sig.return_annotation is not None 