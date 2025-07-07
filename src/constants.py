"""XGBoost Service Constants and Configuration.

This module contains all configuration constants and default parameters used
throughout the MCP XGBoost server application. It defines default XGBoost
parameters, validation limits, supported data formats, and other configuration
values.

Constants
---------
APP_NAME : str
    Application name identifier
USER_AGENT : str
    HTTP user agent string for requests
DEFAULT_XGBOOST_PARAMS : dict
    Default parameters for XGBoost model initialization
MAX_FEATURES : int
    Maximum number of features allowed in datasets
MAX_SAMPLES : int
    Maximum number of samples allowed in datasets
MIN_SAMPLES : int
    Minimum number of samples required for training
SUPPORTED_DATA_FORMATS : list
    List of supported data input formats
MAX_FEATURE_IMPORTANCE_DISPLAY : int
    Maximum number of features to display in importance ranking

Notes
-----
These constants are used throughout the application to ensure consistent
behavior and prevent resource exhaustion. The XGBoost parameters follow
the library's recommended defaults with some modifications for stability.

See Also
--------
xgboost : XGBoost library documentation for parameter explanations
"""

# Application identification constants
APP_NAME = "mcp-xgboost"
"""str: Application name identifier for logging and user agent."""

USER_AGENT = f"{APP_NAME}/1.0"
"""str: HTTP user agent string for external requests."""

# Default XGBoost Parameters
DEFAULT_XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.3,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "random_state": 42,
}
"""dict: Default parameters for XGBoost model initialization.

These parameters provide a good balance between performance and training time
for most use cases. Users can override these when training specific models.

Keys
----
n_estimators : int
    Number of boosting rounds (trees) to train
max_depth : int
    Maximum depth of each tree to prevent overfitting
learning_rate : float
    Step size shrinkage to prevent overfitting
subsample : float
    Fraction of samples used for each tree
colsample_bytree : float
    Fraction of features used for each tree
random_state : int
    Random seed for reproducible results
"""

# Model validation parameters
MAX_FEATURES = 1000
"""int: Maximum number of features allowed in input datasets.

This limit prevents memory issues with very high-dimensional data.
"""

MAX_SAMPLES = 100000
"""int: Maximum number of samples allowed in input datasets.

This limit prevents memory and performance issues with very large datasets.
"""

MIN_SAMPLES = 10
"""int: Minimum number of samples required for model training.

Below this threshold, models may not train reliably.
"""

# Supported data formats
SUPPORTED_DATA_FORMATS = ["json", "csv"]
"""list: List of supported data input formats.

Currently supports JSON and CSV formats for data input.
"""

# Feature importance display limits
MAX_FEATURE_IMPORTANCE_DISPLAY = 20
"""int: Maximum number of features to display in importance ranking.

Limits the output length for feature importance analysis to keep
results manageable and readable.
"""
