"""XGBoost Machine Learning Service Implementation.

This module provides a comprehensive service for XGBoost machine learning
operations through the MCP (Model Context Protocol) framework. It handles
model training, prediction, analysis, and management with full integration
into the MCP server ecosystem.

The service implements robust error handling, data validation, and user-friendly
output formatting suitable for AI assistant integration.

Classes
-------
XGBoostService
    Main service class providing XGBoost ML capabilities through MCP tools

MCP Tools
---------
The service exposes the following MCP tools:
- train_model: Train XGBoost models with custom parameters
- predict: Make predictions using trained models
- get_model_info: Get detailed information about trained models
- get_feature_importance: Analyze feature importance for trained models
- list_models: List all available trained models

Notes
-----
This service is designed to be stateful, maintaining trained models in memory
for the lifetime of the service instance. Models are identified by user-provided
names and can be used for multiple prediction requests.

The service supports both regression and classification tasks, automatically
selecting the appropriate XGBoost model type based on user specification.

See Also
--------
xgboost : XGBoost machine learning library
fastmcp : FastMCP framework for MCP server implementation
src.utils : Utility functions for data validation and formatting
"""

import json
from typing import Any, Dict

import pandas as pd
import xgboost as xgb
from fastmcp import FastMCP

from ..utils import format_feature_importance, format_model_info, validate_data
from ..persistence import ModelNotFoundError


class XGBoostService:
    """XGBoost machine learning service for MCP integration.

    This service provides comprehensive XGBoost functionality through MCP tools,
    enabling AI assistants to train models, make predictions, and analyze results.
    It uses shared storage for model persistence to support distributed deployments.

    Parameters
    ----------
    mcp : FastMCP
        The MCP server instance to register tools with

    Attributes
    ----------
    _mcp : FastMCP
        Reference to the MCP server instance
    _storage : ModelStorage
        Shared storage instance for model persistence

    Methods
    -------
    train_model
        Train an XGBoost model with specified parameters
    predict
        Make predictions using a trained model
    get_model_info
        Get detailed information about a trained model
    get_feature_importance
        Analyze feature importance for a trained model
    list_models
        List all available trained models

    Examples
    --------
    Create and use the service:

        >>> from fastmcp import FastMCP
        >>> mcp = FastMCP("Test Server")
        >>> service = XGBoostService(mcp)
        >>> # Models are now registered as MCP tools

    Notes
    -----
    The service automatically registers all tools with the MCP server during
    initialization. Models are stored in shared storage (file-based or Redis)
    to support distributed deployments in Kubernetes.

    All methods include comprehensive error handling and return user-friendly
    error messages rather than raising exceptions.

    See Also
    --------
    xgboost.XGBRegressor : XGBoost regression model
    xgboost.XGBClassifier : XGBoost classification model
    fastmcp.FastMCP : FastMCP server framework
    src.persistence.ModelStorage : Model storage backend
    """

    def __init__(self, mcp: FastMCP):
        """Initialize the XGBoost service with MCP integration.

        Parameters
        ----------
        mcp : FastMCP
            The MCP server instance to register tools with
        """
        self._mcp = mcp
        # Use shared storage instead of in-memory storage
        from ..persistence import ModelStorage
        self._storage = ModelStorage()
        # Register tools using FastMCP 2.0 decorator syntax
        self._register_tools()

    def _register_tools(self) -> None:
        """Register XGBoost tools with the MCP server.

        This method uses FastMCP's decorator syntax to register all XGBoost
        functionality as MCP tools. Each tool is implemented as a nested
        function that calls the corresponding implementation method.

        Notes
        -----
        This method is called automatically during initialization and should
        not be called directly. The tools are registered with comprehensive
        docstrings that are exposed through the MCP protocol.
        """

        @self._mcp.tool
        async def train_model(  # pylint: disable=too-many-arguments
            model_name: str,
            data: str,
            target_column: str,
            model_type: str = "regression",
            n_estimators: int = 100,
            max_depth: int = 6,
            learning_rate: float = 0.3,
            subsample: float = 1.0,
            colsample_bytree: float = 1.0,
            random_state: int = 42,
        ) -> str:
            """Train an XGBoost model on provided data.

            Parameters
            ----------
            model_name : str
                Name to save the model under for future reference
            data : str
                JSON string containing the training data as a dictionary
                with column names as keys and lists of values
            target_column : str
                Name of the target column in the data
            model_type : str, default="regression"
                Type of model to train ("regression" or "classification")
            n_estimators : int, default=100
                Number of boosting rounds (trees) to train
            max_depth : int, default=6
                Maximum depth of each tree to control overfitting
            learning_rate : float, default=0.3
                Step size shrinkage used to prevent overfitting
            subsample : float, default=1.0
                Fraction of samples to use for each tree
            colsample_bytree : float, default=1.0
                Fraction of features to use for each tree
            random_state : int, default=42
                Random seed for reproducible results

            Returns
            -------
            str
                Success message with training details or error message

            Examples
            --------
            Train a regression model:

                data = '{"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [7, 8, 9]}'
                result = await train_model("my_model", data, "target", "regression")

            Train a classification model:

                data = '{"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}'
                result = await train_model("classifier", data, "target", "classification")
            """
            kwargs = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "random_state": random_state,
            }
            return await self._train_model_impl(
                model_name=model_name, data=data, target_column=target_column, model_type=model_type, **kwargs
            )

        @self._mcp.tool
        async def predict(model_name: str, data: str) -> str:
            """Make predictions using a trained model.

            Parameters
            ----------
            model_name : str
                Name of the trained model to use for predictions
            data : str
                JSON string containing the prediction data as a dictionary
                with the same column structure as the training data (excluding target)

            Returns
            -------
            str
                Formatted prediction results or error message

            Examples
            --------
            Make predictions:

                data = '{"feature1": [1, 2], "feature2": [3, 4]}'
                result = await predict("my_model", data)
            """
            return await self._predict_impl(model_name, data)

        @self._mcp.tool
        async def get_model_info(model_name: str) -> str:
            """Get detailed information about a trained model.

            Parameters
            ----------
            model_name : str
                Name of the model to get information about

            Returns
            -------
            str
                Formatted model information including parameters and status,
                or error message if model not found

            Examples
            --------
            Get model information:

                info = await get_model_info("my_model")
            """
            return await self._get_model_info_impl(model_name)

        @self._mcp.tool
        async def get_feature_importance(model_name: str) -> str:
            """Get feature importance analysis for a trained model.

            Parameters
            ----------
            model_name : str
                Name of the model to analyze

            Returns
            -------
            str
                Formatted feature importance ranking or error message

            Examples
            --------
            Get feature importance:

                importance = await get_feature_importance("my_model")
            """
            return await self._get_feature_importance_impl(model_name)

        @self._mcp.tool
        async def list_models() -> str:
            """List all available trained models.

            Returns
            -------
            str
                List of available models with their types or message
                if no models are available

            Examples
            --------
            List available models:

                models = await list_models()
            """
            return await self._list_models_impl()

    async def _train_model_impl(
        self,
        model_name: str,
        data: str,
        target_column: str,
        model_type: str,
        **kwargs: Any,
    ) -> str:
        """Implementation for training an XGBoost model.

        This method handles the actual model training logic, including data
        validation, model creation, training, and storage.

        Parameters
        ----------
        model_name : str
            Name to save the model under
        data : str
            JSON string containing training data
        target_column : str
            Name of the target column
        model_type : str
            Type of model ("regression" or "classification")
        **kwargs : Any
            Additional parameters for XGBoost model initialization

        Returns
        -------
        str
            Success message with training details or error message

        Notes
        -----
        The method performs comprehensive data validation before training
        and handles all exceptions gracefully, returning user-friendly
        error messages.
        """
        try:
            # Parse the data
            data_dict = json.loads(data)
            df = pd.DataFrame(data_dict)

            # Validate data
            validation_result = validate_data(df, target_column)
            if validation_result != "valid":
                return f"Data validation failed: {validation_result}"

            # Prepare features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Create appropriate model based on type
            if model_type.lower() == "classification":
                model = xgb.XGBClassifier(**kwargs)
            else:
                model = xgb.XGBRegressor(**kwargs)

            # Train the model
            model.fit(X, y)

            # Store the model
            self._storage.save_model(model_name, model, metadata={"description": f"{model_type} model"})

            return (
                f"Model '{model_name}' trained successfully!\n"
                f"Type: {model_type}\n"
                f"Features: {list(X.columns)}\n"
                f"Training samples: {len(X)}"
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return f"Error training model: {str(e)}"

    async def _predict_impl(self, model_name: str, data: str) -> str:
        """Implementation for making predictions with a trained model.

        Parameters
        ----------
        model_name : str
            Name of the model to use for predictions
        data : str
            JSON string containing prediction data

        Returns
        -------
        str
            Formatted prediction results or error message

        Notes
        -----
        The method checks for model existence, validates input data,
        and formats prediction results in a user-friendly manner.
        """
        try:
            # Check if model exists by trying to load it
            try:
                model, metadata = self._storage.load_model(model_name)
            except ModelNotFoundError:
                models = self._storage.list_models()
                available_models = [m.name for m in models]
                return f"Model '{model_name}' not found. Available models: {available_models}"

            # Parse the data
            data_dict = json.loads(data)
            df = pd.DataFrame(data_dict)

            # Make predictions
            predictions = model.predict(df)

            # Format results
            results = []
            for i, pred in enumerate(predictions):
                results.append(f"Sample {i+1}: {pred}")

            return f"Predictions from model '{model_name}':\n" + "\n".join(results)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return f"Error making predictions: {str(e)}"

    async def _get_model_info_impl(self, model_name: str) -> str:
        """Implementation for getting model information.

        Parameters
        ----------
        model_name : str
            Name of the model to get information about

        Returns
        -------
        str
            Formatted model information or error message

        Notes
        -----
        Uses the format_model_info utility function to create
        a comprehensive model information string.
        """
        try:
            # Check if model exists by trying to load it
            try:
                model, metadata = self._storage.load_model(model_name)
            except ModelNotFoundError:
                models = self._storage.list_models()
                available_models = [m.name for m in models]
                return f"Model '{model_name}' not found. Available models: {available_models}"

            return format_model_info(model_name, model)

        except (AttributeError, ValueError) as e:
            return f"Error getting model info: {str(e)}"

    async def _get_feature_importance_impl(self, model_name: str) -> str:
        """Implementation for getting feature importance analysis.

        Parameters
        ----------
        model_name : str
            Name of the model to analyze

        Returns
        -------
        str
            Formatted feature importance ranking or error message

        Notes
        -----
        Uses the format_feature_importance utility function to create
        a ranked list of feature importance values.
        """
        try:
            # Check if model exists by trying to load it
            try:
                model, metadata = self._storage.load_model(model_name)
            except ModelNotFoundError:
                models = self._storage.list_models()
                available_models = [m.name for m in models]
                return f"Model '{model_name}' not found. Available models: {available_models}"

            return format_feature_importance(model)

        except (AttributeError, ValueError) as e:
            return f"Error getting feature importance: {str(e)}"

    async def _list_models_impl(self) -> str:
        """Implementation for listing available models.

        Returns
        -------
        str
            List of available models with their types or message
            if no models are available

        Notes
        -----
        Uses the storage backend to list models and automatically detects 
        model types from metadata or by loading the model.
        """
        try:
            models = self._storage.list_models()
            if not models:
                return "No models have been trained yet."

            model_list = []
            for model_metadata in models:
                model_name = model_metadata.name
                model_type = model_metadata.model_type.title()
                model_list.append(f"- {model_name} ({model_type})")

            return "Available models:\n" + "\n".join(model_list)
        
        except Exception as e:
            return f"Error listing models: {str(e)}"

    @property
    def mcp(self) -> FastMCP:
        """Get the MCP server instance.

        Returns
        -------
        FastMCP
            The MCP server instance used by this service

        Notes
        -----
        This property provides access to the underlying MCP server
        for advanced use cases or debugging.
        """
        return self._mcp
