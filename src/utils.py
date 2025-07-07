"""Utility Functions for XGBoost Operations.

This module provides utility functions for data validation, model information
formatting, feature importance analysis, and sample data generation. These
functions support the core XGBoost service operations and help maintain
consistency across the application.

Functions
---------
validate_data
    Validate input data for XGBoost model training
format_model_info
    Format model information into a readable string
format_feature_importance
    Format feature importance information into a readable string
prepare_sample_data
    Generate sample data for XGBoost demonstration

Notes
-----
All functions in this module are designed to be robust and handle edge cases
gracefully, returning informative error messages when issues are encountered.

See Also
--------
pandas : Data manipulation and analysis library
xgboost : XGBoost machine learning library
numpy : Numerical computing library
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import xgboost as xgb


def validate_data(df: pd.DataFrame, target_column: str) -> str:
    """Validate input data for XGBoost model training.

    Performs comprehensive validation of input data including checks for
    empty datasets, missing target columns, null values, and data type
    compatibility with XGBoost.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset to validate
    target_column : str
        Name of the target column for model training

    Returns
    -------
    str
        Validation result message. Returns "valid" if data passes all checks,
        otherwise returns a descriptive error message explaining the issue.

    Examples
    --------
    Validate a simple dataset:

        >>> import pandas as pd
        >>> df = pd.DataFrame({'feature1': [1, 2, 3], 'target': [0, 1, 0]})
        >>> result = validate_data(df, 'target')
        >>> print(result)
        'valid'

    Validate dataset with missing target:

        >>> df = pd.DataFrame({'feature1': [1, 2, 3]})
        >>> result = validate_data(df, 'target')
        >>> print(result)
        "Target column 'target' not found in data"

    Notes
    -----
    The function checks for:

    - Empty datasets
    - Missing target columns
    - Null values in target column
    - Excessive null values (>50%) in any column
    - Non-numeric columns that may need preprocessing

    See Also
    --------
    pandas.DataFrame.isnull : Check for null values
    pandas.DataFrame.select_dtypes : Select columns by data type
    """
    if df.empty:
        return "Dataset is empty"

    if target_column not in df.columns:
        return f"Target column '{target_column}' not found in data"

    if df[target_column].isnull().all():
        return f"Target column '{target_column}' contains only null values"

    # Check for excessive null values
    null_percentage = df.isnull().sum() / len(df) * 100
    high_null_cols = null_percentage[null_percentage > 50].index.tolist()
    if high_null_cols:
        return f"Columns with >50% null values: {high_null_cols}"

    # Check for non-numeric data that might need preprocessing
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    if target_column in non_numeric_cols:
        non_numeric_cols.remove(target_column)

    if non_numeric_cols:
        return f"Non-numeric columns detected (may need encoding): {non_numeric_cols}"

    return "valid"


def format_model_info(model_name: str, model: xgb.XGBModel) -> str:
    """Format model information into a readable string.

    Extracts and formats key information about a trained XGBoost model,
    including model type, training status, and important hyperparameters.

    Parameters
    ----------
    model_name : str
        Name identifier for the model
    model : xgb.XGBModel
        Trained XGBoost model instance (XGBRegressor or XGBClassifier)

    Returns
    -------
    str
        Formatted string containing model information including:

        - Model name and type (Classification/Regression)
        - Training status
        - Key hyperparameters (n_estimators, max_depth, etc.)
        - Ready status for predictions

    Examples
    --------
    Format information for a regression model:

        >>> import xgboost as xgb
        >>> model = xgb.XGBRegressor(n_estimators=100, max_depth=6)
        >>> # ... train the model ...
        >>> info = format_model_info("my_model", model)
        >>> print(info)
        Model: my_model
        Type: Regression
        Status: Trained
        ...

    Notes
    -----
    The function automatically detects model type by checking for the
    predict_proba method, which is only available in classification models.

    See Also
    --------
    xgboost.XGBModel.get_params : Get model parameters
    """
    model_type = "Classification" if hasattr(model, "predict_proba") else "Regression"

    # Get model parameters
    params = model.get_params()

    # Format key parameters
    key_params = {
        "n_estimators": params.get("n_estimators", "Not set"),
        "max_depth": params.get("max_depth", "Not set"),
        "learning_rate": params.get("learning_rate", "Not set"),
        "subsample": params.get("subsample", "Not set"),
        "colsample_bytree": params.get("colsample_bytree", "Not set"),
        "objective": params.get("objective", "Not set"),
    }

    info = f"""
Model: {model_name}
Type: {model_type}
Status: Trained

Key Parameters:
- Number of estimators: {key_params['n_estimators']}
- Max depth: {key_params['max_depth']}
- Learning rate: {key_params['learning_rate']}
- Subsample: {key_params['subsample']}
- Column sample by tree: {key_params['colsample_bytree']}
- Objective: {key_params['objective']}

Model is ready for predictions and analysis.
"""

    return info.strip()


def format_feature_importance(model: xgb.XGBModel) -> str:
    """Format feature importance information into a readable string.

    Extracts feature importance values from a trained XGBoost model and
    formats them into a ranked, human-readable string. Limits output to
    the top 20 features for readability.

    Parameters
    ----------
    model : xgb.XGBModel
        Trained XGBoost model instance with feature importance data

    Returns
    -------
    str
        Formatted string containing:

        - Ranked list of features by importance
        - Importance values (4 decimal places)
        - Indication of additional features if more than 20 exist

    Examples
    --------
    Format feature importance for a trained model:

        >>> import xgboost as xgb
        >>> import pandas as pd
        >>> X = pd.DataFrame({'feat1': [1, 2], 'feat2': [3, 4]})
        >>> y = [0, 1]
        >>> model = xgb.XGBRegressor()
        >>> model.fit(X, y)
        >>> importance = format_feature_importance(model)
        >>> print(importance)
        Feature Importance (sorted by importance):
         1. feat1: 0.5000
         2. feat2: 0.5000

    Notes
    -----
    Feature names are obtained from the model's feature_names_in_ attribute
    if available, otherwise generic names (Feature_0, Feature_1, etc.) are used.

    The function handles errors gracefully and returns an error message if
    feature importance cannot be extracted.

    Raises
    ------
    The function catches AttributeError and ValueError exceptions and returns
    them as formatted error messages rather than raising them.

    See Also
    --------
    xgboost.XGBModel.feature_importances_ : Get feature importance values
    src.constants.MAX_FEATURE_IMPORTANCE_DISPLAY : Maximum features to display
    """
    try:
        # Get feature importance
        importance = model.feature_importances_

        # Get feature names (if available)
        if hasattr(model, "feature_names_in_"):
            feature_names = model.feature_names_in_
        else:
            feature_names = [f"Feature_{i}" for i in range(len(importance))]

        # Sort by importance
        feature_importance = list(zip(feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        # Format output
        importance_text = "Feature Importance (sorted by importance):\n"
        # Top 20 features
        for i, (feature, imp) in enumerate(feature_importance[:20]):
            importance_text += f"{i+1:2d}. {feature}: {imp:.4f}\n"

        if len(feature_importance) > 20:
            importance_text += f"... and {len(feature_importance) - 20} more features"

        return importance_text

    except (AttributeError, ValueError) as e:
        return f"Error getting feature importance: {str(e)}"


def prepare_sample_data() -> Dict[str, Any]:
    """Generate sample data for XGBoost demonstration.

    Creates a synthetic regression dataset with known relationships between
    features and target variable. This data is useful for testing,
    demonstrations, and examples.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing sample data with keys:

        - 'feature1', 'feature2', 'feature3': List of feature values
        - 'target': List of target values with known relationship to features

    Examples
    --------
    Generate sample data for training:

        >>> data = prepare_sample_data()
        >>> print(data.keys())
        dict_keys(['feature1', 'feature2', 'feature3', 'target'])
        >>> len(data['feature1'])
        100

    Use sample data with pandas DataFrame:

        >>> import pandas as pd
        >>> data = prepare_sample_data()
        >>> df = pd.DataFrame(data)
        >>> print(df.shape)
        (100, 4)

    Notes
    -----
    The target variable is generated using the relationship:
    target = 2 * feature1 + 1.5 * feature2 - 0.5 * feature3 + noise

    This provides a realistic regression scenario where the model can learn
    the underlying relationships. Random seed is set to 42 for reproducibility.

    The dataset contains 100 samples with 3 features and 1 target variable.
    All features are generated from a standard normal distribution.

    See Also
    --------
    numpy.random.seed : Set random seed for reproducibility
    numpy.random.randn : Generate random numbers from standard normal distribution
    """
    # Create a simple regression dataset
    np.random.seed(42)
    n_samples = 100

    # Generate features
    feature1 = np.random.randn(n_samples)
    feature2 = np.random.randn(n_samples)
    feature3 = np.random.randn(n_samples)

    # Generate target with some relationship to features
    target = 2 * feature1 + 1.5 * feature2 - 0.5 * feature3 + np.random.randn(n_samples) * 0.1

    return {
        "feature1": feature1.tolist(),
        "feature2": feature2.tolist(),
        "feature3": feature3.tolist(),
        "target": target.tolist(),
    }
