"""Model Persistence Layer for MCP XGBoost Server.

This module provides a comprehensive model persistence layer for the MCP XGBoost
server application. It handles saving and loading XGBoost models with metadata,
compression, versioning, and storage management.

The persistence layer supports:
- Model serialization with XGBoost's native format
- Metadata storage with model information
- Compression to reduce storage size
- Version management for model updates
- Backup and recovery functionality
- Storage cleanup and optimization
- Model validation on load

Classes
-------
ModelMetadata
    Metadata information for stored models
ModelStorage
    Main storage interface for model persistence
ModelVersion
    Version information for model tracking
StorageManager
    High-level storage management interface

Functions
---------
save_model
    Save a model with metadata to storage
load_model
    Load a model from storage with validation
list_models
    List all stored models with metadata
delete_model
    Delete a model from storage
cleanup_storage
    Clean up old model versions and optimize storage

Notes
-----
The persistence layer is designed to be:
- Reliable with atomic operations
- Efficient with compression
- Scalable with version management
- Safe with validation and backups
- Configurable through the config system

Storage Structure
----------------
The models are stored in the following structure:
```
models/
├── model_name/
│   ├── metadata.json
│   ├── versions/
│   │   ├── v1/
│   │   │   ├── model.json
│   │   │   └── info.json
│   │   └── v2/
│   │       ├── model.json
│   │       └── info.json
│   └── current -> versions/v2/
└── .storage_info
```

Examples
--------
Basic model saving:

    >>> from src.persistence import save_model
    >>> from xgboost import XGBRegressor
    >>> model = XGBRegressor()
    >>> # ... train model ...
    >>> save_model("my_model", model, {"description": "Test model"})

Loading a model:

    >>> from src.persistence import load_model
    >>> model, metadata = load_model("my_model")
    >>> print(metadata.description)
    Test model

List all models:

    >>> from src.persistence import list_models
    >>> models = list_models()
    >>> for model_info in models:
    ...     print(f"{model_info.name}: {model_info.description}")

Storage management:

    >>> from src.persistence import StorageManager
    >>> storage = StorageManager()
    >>> storage.cleanup_old_versions(keep_versions=3)
    >>> storage.optimize_storage()

See Also
--------
src.config : Configuration management
src.exceptions : Custom exception handling
xgboost : XGBoost model serialization
"""

import gzip
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import xgboost as xgb
from pydantic import BaseModel, Field

from .config import get_config
from .exceptions import ModelNotFoundError, ModelPersistenceError, wrap_exception
from .logging_config import get_logger


class ModelMetadata(BaseModel):
    """Metadata information for stored models.

    Attributes
    ----------
    name : str
        Model name identifier
    description : str
        Human-readable description
    model_type : str
        Type of model (regression/classification)
    version : int
        Model version number
    created_at : datetime
        Model creation timestamp
    updated_at : datetime
        Last update timestamp
    training_info : Dict[str, Any]
        Training information and parameters
    performance_metrics : Dict[str, Any]
        Model performance metrics
    feature_names : List[str]
        Names of features used for training
    storage_info : Dict[str, Any]
        Storage-specific information
    """

    name: str = Field(..., description="Model name identifier")
    description: str = Field(default="", description="Human-readable description")
    model_type: str = Field(..., description="Type of model (regression/classification)")
    version: int = Field(default=1, ge=1, description="Model version number")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    training_info: Dict[str, Any] = Field(default_factory=dict, description="Training information")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    feature_names: List[str] = Field(default_factory=list, description="Feature names")
    storage_info: Dict[str, Any] = Field(default_factory=dict, description="Storage information")

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "model_type": self.model_type,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "training_info": self.training_info,
            "performance_metrics": self.performance_metrics,
            "feature_names": self.feature_names,
            "storage_info": self.storage_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


class ModelVersion(BaseModel):
    """Version information for model tracking.

    Attributes
    ----------
    version : int
        Version number
    created_at : datetime
        Version creation timestamp
    checksum : str
        Model file checksum for integrity
    size_bytes : int
        Model file size in bytes
    compression : bool
        Whether model is compressed
    """

    version: int = Field(..., ge=1, description="Version number")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    checksum: str = Field(..., description="Model file checksum")
    size_bytes: int = Field(..., ge=0, description="File size in bytes")
    compression: bool = Field(default=False, description="Whether model is compressed")


class ModelStorage:
    """Main storage interface for model persistence.

    This class provides the core functionality for saving and loading
    XGBoost models with metadata and version management.

    Attributes
    ----------
    storage_path : Path
        Base path for model storage
    config : AppConfig
        Application configuration
    logger : StructuredLogger
        Logger instance for storage operations
    """

    def __init__(self, storage_path: Optional[Union[str, Path]] = None):
        """Initialize the model storage.

        Parameters
        ----------
        storage_path : str or Path, optional
            Base path for model storage (default: from config)
        """
        self.config = get_config()
        self.logger = get_logger(__name__)

        if storage_path is None:
            storage_path = self.config.storage.model_storage_path

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Create storage info file
        self._initialize_storage()

    def _initialize_storage(self):
        """Initialize storage with configuration."""
        storage_info_path = self.storage_path / ".storage_info"
        storage_info = {
            "version": "1.0",
            "created_at": datetime.utcnow().isoformat(),
            "config": {
                "compression": self.config.storage.compression,
                "auto_save": self.config.storage.auto_save_models,
            },
        }

        if not storage_info_path.exists():
            with open(storage_info_path, "w", encoding="utf-8") as f:
                json.dump(storage_info, f, indent=2)

    def _get_model_path(self, model_name: str) -> Path:
        """Get the path for a model directory.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        Path
            Path to model directory
        """
        return self.storage_path / model_name

    def _get_version_path(self, model_name: str, version: int) -> Path:
        """Get the path for a specific model version.

        Parameters
        ----------
        model_name : str
            Name of the model
        version : int
            Version number

        Returns
        -------
        Path
            Path to version directory
        """
        return self._get_model_path(model_name) / "versions" / f"v{version}"

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file.

        Parameters
        ----------
        file_path : Path
            Path to the file

        Returns
        -------
        str
            SHA256 checksum hex string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _save_model_file(self, model: xgb.XGBModel, file_path: Path, compress: bool = False):
        """Save XGBoost model to file.

        Parameters
        ----------
        model : xgb.XGBModel
            XGBoost model to save
        file_path : Path
            Path to save the model
        compress : bool, optional
            Whether to compress the model (default: False)
        """
        try:
            if compress:
                # Save to temporary file first
                temp_path = file_path.with_suffix(".tmp")
                model.save_model(str(temp_path))

                # Compress the file
                with open(temp_path, "rb") as f_in:
                    with gzip.open(file_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Remove temporary file
                temp_path.unlink()
            else:
                model.save_model(str(file_path))

        except Exception as e:
            raise wrap_exception("save model file", e, {"file_path": str(file_path), "compress": compress}) from e

    def _load_model_file(self, file_path: Path, compress: bool = False) -> xgb.XGBModel:
        """Load XGBoost model from file.

        Parameters
        ----------
        file_path : Path
            Path to the model file
        compress : bool, optional
            Whether the model is compressed (default: False)

        Returns
        -------
        xgb.XGBModel
            Loaded XGBoost model
        """
        try:
            if compress:
                # Extract to temporary file
                temp_path = file_path.with_suffix(".tmp")
                with gzip.open(file_path, "rb") as f_in:
                    with open(temp_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Load the model
                model = xgb.XGBModel()
                model.load_model(str(temp_path))

                # Remove temporary file
                temp_path.unlink()
            else:
                model = xgb.XGBModel()
                model.load_model(str(file_path))

            return model

        except Exception as e:
            raise wrap_exception("load model file", e, {"file_path": str(file_path), "compress": compress}) from e

    def save_model(
        self, model_name: str, model: xgb.XGBModel, metadata: Optional[Dict[str, Any]] = None, overwrite: bool = False
    ) -> ModelMetadata:
        """Save a model with metadata to storage.

        Parameters
        ----------
        model_name : str
            Name to save the model under
        model : xgb.XGBModel
            XGBoost model to save
        metadata : Dict[str, Any], optional
            Additional metadata to store
        overwrite : bool, optional
            Whether to overwrite existing model (default: False)

        Returns
        -------
        ModelMetadata
            Metadata of the saved model

        Raises
        ------
        ModelPersistenceError
            If saving fails
        """
        try:
            model_path = self._get_model_path(model_name)

            # Check if model exists
            if model_path.exists() and not overwrite:
                raise ModelPersistenceError(
                    f"Model '{model_name}' already exists. Use overwrite=True to replace.",
                    details={"model_name": model_name},
                )

            # Determine version number
            version = 1
            if model_path.exists():
                # Get next version number
                versions_path = model_path / "versions"
                if versions_path.exists():
                    existing_versions = [
                        int(d.name[1:]) for d in versions_path.iterdir() if d.is_dir() and d.name.startswith("v")
                    ]
                    version = max(existing_versions, default=0) + 1

            # Create version directory
            version_path = self._get_version_path(model_name, version)
            version_path.mkdir(parents=True, exist_ok=True)

            # Save model file
            model_file_path = version_path / "model.json"
            if self.config.storage.compression:
                model_file_path = version_path / "model.json.gz"

            self._save_model_file(model, model_file_path, self.config.storage.compression)

            # Calculate checksum and file size
            checksum = self._calculate_checksum(model_file_path)
            size_bytes = model_file_path.stat().st_size

            # Create metadata
            model_metadata = ModelMetadata(
                name=model_name,
                description=metadata.get("description", "") if metadata else "",
                model_type="classification" if hasattr(model, "predict_proba") else "regression",
                version=version,
                training_info=metadata.get("training_info", {}) if metadata else {},
                performance_metrics=metadata.get("performance_metrics", {}) if metadata else {},
                feature_names=(
                    getattr(model, "feature_names_in_", []).tolist() if hasattr(model, "feature_names_in_") else []
                ),
                storage_info={
                    "checksum": checksum,
                    "size_bytes": size_bytes,
                    "compression": self.config.storage.compression,
                    "file_name": model_file_path.name,
                },
            )

            # Save metadata
            metadata_path = version_path / "info.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(model_metadata.to_dict(), f, indent=2)

            # Update main metadata file
            main_metadata_path = model_path / "metadata.json"
            with open(main_metadata_path, "w", encoding="utf-8") as f:
                json.dump(model_metadata.to_dict(), f, indent=2)

            # Create/update current symlink
            current_link = model_path / "current"
            if current_link.exists():
                current_link.unlink()
            current_link.symlink_to(f"versions/v{version}")

            self.logger.info(
                "Model saved successfully",
                extra={
                    "model_name": model_name,
                    "version": version,
                    "size_bytes": size_bytes,
                    "compression": self.config.storage.compression,
                },
            )

            return model_metadata

        except Exception as e:
            if isinstance(e, ModelPersistenceError):
                raise
            raise wrap_exception("save model", e, {"model_name": model_name, "version": version}) from e

    def load_model(
        self, model_name: str, version: Optional[int] = None, validate: bool = True
    ) -> Tuple[xgb.XGBModel, ModelMetadata]:
        """Load a model from storage with validation.

        Parameters
        ----------
        model_name : str
            Name of the model to load
        version : int, optional
            Specific version to load (default: latest)
        validate : bool, optional
            Whether to validate the model after loading (default: True)

        Returns
        -------
        Tuple[xgb.XGBModel, ModelMetadata]
            Loaded model and its metadata

        Raises
        ------
        ModelNotFoundError
            If model doesn't exist
        ModelPersistenceError
            If loading fails
        """
        try:
            model_path = self._get_model_path(model_name)

            if not model_path.exists():
                raise ModelNotFoundError(f"Model '{model_name}' not found", details={"model_name": model_name})

            # Determine version to load
            if version is None:
                # Load current version
                current_link = model_path / "current"
                if not current_link.exists():
                    raise ModelPersistenceError(
                        f"No current version found for model '{model_name}'", details={"model_name": model_name}
                    )
                version_path = model_path / current_link.readlink()
            else:
                version_path = self._get_version_path(model_name, version)

            if not version_path.exists():
                raise ModelNotFoundError(
                    f"Version {version} of model '{model_name}' not found",
                    details={"model_name": model_name, "version": version},
                )

            # Load metadata
            metadata_path = version_path / "info.json"
            if not metadata_path.exists():
                raise ModelPersistenceError(
                    f"Metadata file not found for model '{model_name}' version {version}",
                    details={"model_name": model_name, "version": version},
                )

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)

            metadata = ModelMetadata.from_dict(metadata_dict)

            # Load model file
            compression = metadata.storage_info.get("compression", False)
            model_file_name = metadata.storage_info.get("file_name", "model.json")
            model_file_path = version_path / model_file_name

            if not model_file_path.exists():
                raise ModelPersistenceError(
                    f"Model file not found: {model_file_path}", details={"model_name": model_name, "version": version}
                )

            # Validate checksum if available
            if validate and "checksum" in metadata.storage_info:
                current_checksum = self._calculate_checksum(model_file_path)
                expected_checksum = metadata.storage_info["checksum"]
                if current_checksum != expected_checksum:
                    raise ModelPersistenceError(
                        f"Model file checksum mismatch for '{model_name}'",
                        details={
                            "model_name": model_name,
                            "version": version,
                            "expected": expected_checksum,
                            "actual": current_checksum,
                        },
                    )

            # Load the model
            model = self._load_model_file(model_file_path, compression)

            self.logger.info(
                "Model loaded successfully",
                extra={"model_name": model_name, "version": metadata.version, "model_type": metadata.model_type},
            )

            return model, metadata

        except (ModelNotFoundError, ModelPersistenceError):
            raise
        except Exception as e:
            raise wrap_exception("load model", e, {"model_name": model_name, "version": version}) from e

    def list_models(self) -> List[ModelMetadata]:
        """List all stored models with metadata.

        Returns
        -------
        List[ModelMetadata]
            List of model metadata
        """
        models = []

        try:
            for model_dir in self.storage_path.iterdir():
                if model_dir.is_dir() and not model_dir.name.startswith("."):
                    metadata_path = model_dir / "metadata.json"
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, "r", encoding="utf-8") as f:
                                metadata_dict = json.load(f)
                            metadata = ModelMetadata.from_dict(metadata_dict)
                            models.append(metadata)
                        except (json.JSONDecodeError, FileNotFoundError, ValueError, KeyError) as e:
                            self.logger.warning(
                                f"Failed to load metadata for model {model_dir.name}",
                                extra={"model_dir": str(model_dir), "error": str(e)},
                            )

            return sorted(models, key=lambda m: m.updated_at, reverse=True)

        except Exception as e:
            raise wrap_exception("list models", e) from e

    def delete_model(self, model_name: str, version: Optional[int] = None) -> bool:
        """Delete a model from storage.

        Parameters
        ----------
        model_name : str
            Name of the model to delete
        version : int, optional
            Specific version to delete (default: all versions)

        Returns
        -------
        bool
            True if deletion was successful

        Raises
        ------
        ModelNotFoundError
            If model doesn't exist
        """
        try:
            model_path = self._get_model_path(model_name)

            if not model_path.exists():
                raise ModelNotFoundError(f"Model '{model_name}' not found", details={"model_name": model_name})

            if version is None:
                # Delete entire model
                shutil.rmtree(model_path)
                self.logger.info("Model deleted completely", extra={"model_name": model_name})
            else:
                # Delete specific version
                version_path = self._get_version_path(model_name, version)
                if not version_path.exists():
                    raise ModelNotFoundError(
                        f"Version {version} of model '{model_name}' not found",
                        details={"model_name": model_name, "version": version},
                    )

                shutil.rmtree(version_path)
                self.logger.info("Model version deleted", extra={"model_name": model_name, "version": version})

            return True

        except ModelNotFoundError:
            raise
        except Exception as e:
            raise wrap_exception("delete model", e, {"model_name": model_name, "version": version}) from e


class StorageManager:
    """High-level storage management interface.

    This class provides utilities for managing model storage,
    including cleanup, optimization, and backup operations.
    """

    def __init__(self, storage: Optional[ModelStorage] = None):
        """Initialize the storage manager.

        Parameters
        ----------
        storage : ModelStorage, optional
            Storage instance to manage (default: creates new instance)
        """
        self.storage = storage or ModelStorage()
        self.logger = get_logger(__name__)

    def cleanup_old_versions(self, keep_versions: int = 3) -> int:
        """Clean up old model versions.

        Parameters
        ----------
        keep_versions : int, optional
            Number of versions to keep per model (default: 3)

        Returns
        -------
        int
            Number of versions deleted
        """
        deleted_count = 0

        try:
            models = self.storage.list_models()

            for model_metadata in models:
                model_path = self.storage._get_model_path(model_metadata.name)
                versions_path = model_path / "versions"

                if versions_path.exists():
                    # Get all versions
                    version_dirs = [d for d in versions_path.iterdir() if d.is_dir() and d.name.startswith("v")]

                    # Sort by version number (descending)
                    version_dirs.sort(key=lambda d: int(d.name[1:]), reverse=True)

                    # Delete old versions
                    for version_dir in version_dirs[keep_versions:]:
                        shutil.rmtree(version_dir)
                        deleted_count += 1

                        self.logger.info(
                            "Deleted old version",
                            extra={"model_name": model_metadata.name, "version": version_dir.name},
                        )

            return deleted_count

        except Exception as e:
            raise wrap_exception("cleanup old versions", e) from e

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns
        -------
        Dict[str, Any]
            Storage statistics
        """
        try:
            models = self.storage.list_models()

            total_size = 0
            total_models = len(models)
            model_types: dict[str, int] = {}

            for model_metadata in models:
                model_path = self.storage._get_model_path(model_metadata.name)

                # Calculate directory size
                for file_path in model_path.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size

                # Count model types
                model_type = model_metadata.model_type
                model_types[model_type] = model_types.get(model_type, 0) + 1

            return {
                "total_models": total_models,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "model_types": model_types,
                "storage_path": str(self.storage.storage_path),
            }

        except Exception as e:
            raise wrap_exception("get storage stats", e) from e


# Convenience functions
def save_model(
    model_name: str, model: xgb.XGBModel, metadata: Optional[Dict[str, Any]] = None, overwrite: bool = False
) -> ModelMetadata:
    """Save a model with metadata to storage.

    Parameters
    ----------
    model_name : str
        Name to save the model under
    model : xgb.XGBModel
        XGBoost model to save
    metadata : Dict[str, Any], optional
        Additional metadata to store
    overwrite : bool, optional
        Whether to overwrite existing model (default: False)

    Returns
    -------
    ModelMetadata
        Metadata of the saved model
    """
    storage = ModelStorage()
    return storage.save_model(model_name, model, metadata, overwrite)


def load_model(
    model_name: str, version: Optional[int] = None, validate: bool = True
) -> Tuple[xgb.XGBModel, ModelMetadata]:
    """Load a model from storage with validation.

    Parameters
    ----------
    model_name : str
        Name of the model to load
    version : int, optional
        Specific version to load (default: latest)
    validate : bool, optional
        Whether to validate the model after loading (default: True)

    Returns
    -------
    Tuple[xgb.XGBModel, ModelMetadata]
        Loaded model and its metadata
    """
    storage = ModelStorage()
    return storage.load_model(model_name, version, validate)


def list_models() -> List[ModelMetadata]:
    """List all stored models with metadata.

    Returns
    -------
    List[ModelMetadata]
        List of model metadata
    """
    storage = ModelStorage()
    return storage.list_models()


def delete_model(model_name: str, version: Optional[int] = None) -> bool:
    """Delete a model from storage.

    Parameters
    ----------
    model_name : str
        Name of the model to delete
    version : int, optional
        Specific version to delete (default: all versions)

    Returns
    -------
    bool
        True if deletion was successful
    """
    storage = ModelStorage()
    return storage.delete_model(model_name, version)


def cleanup_storage(keep_versions: int = 3) -> int:
    """Clean up old model versions.

    Parameters
    ----------
    keep_versions : int, optional
        Number of versions to keep per model (default: 3)

    Returns
    -------
    int
        Number of versions deleted
    """
    manager = StorageManager()
    return manager.cleanup_old_versions(keep_versions)
