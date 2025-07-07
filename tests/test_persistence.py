"""Test the persistence module comprehensively.

This test module provides comprehensive testing for the persistence layer
to increase coverage from 26% to 70% target.
"""

import gzip
import json
import os
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open

import pytest
import xgboost as xgb
import numpy as np
from pydantic import ValidationError

from src.persistence import (
    ModelMetadata,
    ModelVersion,
    ModelStorage,
    StorageManager,
    save_model,
    load_model,
    list_models,
    delete_model,
    cleanup_storage,
)
from src.exceptions import ModelNotFoundError, ModelPersistenceError
from src.config import AppConfig


class TestModelMetadata:
    """Test ModelMetadata class."""

    def test_model_metadata_creation(self):
        """Test basic model metadata creation."""
        metadata = ModelMetadata(
            name="test_model",
            model_type="regression",
            description="Test model for regression"
        )
        
        assert metadata.name == "test_model"
        assert metadata.model_type == "regression"
        assert metadata.description == "Test model for regression"
        assert metadata.version == 1
        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.updated_at, datetime)

    def test_model_metadata_to_dict(self):
        """Test metadata to dictionary conversion."""
        metadata = ModelMetadata(
            name="test_model",
            model_type="classification",
            training_info={"samples": 1000, "features": 5},
            performance_metrics={"accuracy": 0.95},
            feature_names=["f1", "f2", "f3", "f4", "f5"]
        )
        
        result = metadata.to_dict()
        
        assert isinstance(result, dict)
        assert result["name"] == "test_model"
        assert result["model_type"] == "classification"
        assert result["training_info"]["samples"] == 1000
        assert result["performance_metrics"]["accuracy"] == 0.95
        assert result["feature_names"] == ["f1", "f2", "f3", "f4", "f5"]
        assert "created_at" in result
        assert "updated_at" in result

    def test_model_metadata_from_dict(self):
        """Test metadata from dictionary creation."""
        data = {
            "name": "test_model",
            "model_type": "regression",
            "description": "Test model",
            "version": 2,
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-01T01:00:00",
            "training_info": {"samples": 500},
            "performance_metrics": {"rmse": 0.1},
            "feature_names": ["feature1", "feature2"],
            "storage_info": {"compressed": True}
        }
        
        metadata = ModelMetadata.from_dict(data)
        
        assert metadata.name == "test_model"
        assert metadata.model_type == "regression"
        assert metadata.version == 2
        assert metadata.training_info["samples"] == 500
        assert metadata.performance_metrics["rmse"] == 0.1
        assert metadata.feature_names == ["feature1", "feature2"]
        assert metadata.storage_info["compressed"] is True


class TestModelVersion:
    """Test ModelVersion class."""

    def test_model_version_creation(self):
        """Test model version creation."""
        version = ModelVersion(
            version=1,
            checksum="abc123",
            size_bytes=1024,
            compression=True
        )
        
        assert version.version == 1
        assert version.checksum == "abc123"
        assert version.size_bytes == 1024
        assert version.compression is True
        assert isinstance(version.created_at, datetime)

    def test_model_version_validation(self):
        """Test model version validation."""
        # Test invalid version number
        with pytest.raises(ValidationError):
            ModelVersion(version=0, checksum="abc", size_bytes=100)
        
        # Test invalid size
        with pytest.raises(ValidationError):
            ModelVersion(version=1, checksum="abc", size_bytes=-1)


class TestModelStorage:
    """Test ModelStorage class."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "models"
        
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.storage.model_storage_path = str(self.storage_path)
        self.mock_config.storage.compression = False
        self.mock_config.storage.auto_save_models = True

    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.persistence.get_config')
    def test_model_storage_initialization(self, mock_get_config):
        """Test ModelStorage initialization."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        
        assert storage.storage_path.exists()
        assert (storage.storage_path / ".storage_info").exists()

    @patch('src.persistence.get_config')
    def test_model_storage_with_custom_path(self, mock_get_config):
        """Test ModelStorage with custom path."""
        mock_get_config.return_value = self.mock_config
        
        custom_path = Path(self.temp_dir) / "custom_models"
        storage = ModelStorage(custom_path)
        
        assert storage.storage_path == custom_path
        assert custom_path.exists()

    @patch('src.persistence.get_config')
    def test_get_model_path(self, mock_get_config):
        """Test _get_model_path method."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        model_path = storage._get_model_path("test_model")
        
        assert model_path == storage.storage_path / "test_model"

    @patch('src.persistence.get_config')
    def test_get_version_path(self, mock_get_config):
        """Test _get_version_path method."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        version_path = storage._get_version_path("test_model", 1)
        
        assert version_path == storage.storage_path / "test_model" / "versions" / "v1"

    @patch('src.persistence.get_config')
    def test_calculate_checksum(self, mock_get_config):
        """Test _calculate_checksum method."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        checksum = storage._calculate_checksum(test_file)
        
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex length

    @patch('src.persistence.get_config')
    def test_save_model_file_uncompressed(self, mock_get_config):
        """Test _save_model_file method without compression."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        
        # Create mock XGBoost model
        mock_model = Mock()
        mock_model.save_model = Mock()
        
        file_path = Path(self.temp_dir) / "model.json"
        storage._save_model_file(mock_model, file_path, compress=False)
        
        mock_model.save_model.assert_called_once_with(str(file_path))

    @patch('src.persistence.get_config')
    def test_save_model_file_compressed(self, mock_get_config):
        """Test _save_model_file method with compression."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        
        # Create mock XGBoost model
        mock_model = Mock()
        mock_model.save_model = Mock()
        
        # Create a temporary file to simulate model save
        temp_file = Path(self.temp_dir) / "model.json.tmp"
        temp_file.write_text("test model content")
        
        with patch('builtins.open', mock_open()) as mock_file_open:
            with patch('gzip.open', mock_open()) as mock_gzip_open:
                file_path = Path(self.temp_dir) / "model.json.gz"
                storage._save_model_file(mock_model, file_path, compress=True)
                
                # Should call save_model with temp path
                mock_model.save_model.assert_called_once()

    @patch('src.persistence.get_config')
    def test_load_model_file_uncompressed(self, mock_get_config):
        """Test _load_model_file method without compression."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        
        # Create test file
        test_file = Path(self.temp_dir) / "model.json"
        test_file.write_text("test model")
        
        with patch('xgboost.XGBModel') as mock_xgb_model:
            mock_instance = Mock()
            mock_xgb_model.return_value = mock_instance
            
            result = storage._load_model_file(test_file, compress=False)
            
            assert result == mock_instance
            mock_instance.load_model.assert_called_once_with(str(test_file))

    @patch('src.persistence.get_config')
    def test_save_model_new_model(self, mock_get_config):
        """Test save_model method with new model."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        
        # Create mock XGBoost model
        mock_model = Mock()
        mock_model.save_model = Mock()
        
        # Mock hasattr to return False for predict_proba (regression model)
        with patch('builtins.hasattr', return_value=False):
            with patch.object(storage, '_save_model_file') as mock_save_file:
                with patch.object(storage, '_calculate_checksum', return_value="test_checksum"):
                    with patch('pathlib.Path.stat') as mock_stat:
                        with patch('pathlib.Path.exists', return_value=False):  # Model doesn't exist yet
                            with patch('pathlib.Path.symlink_to') as mock_symlink:  # Mock symlink creation
                                mock_stat.return_value.st_size = 1024
                                metadata = storage.save_model(
                                    "test_model",
                                    mock_model,
                                    {"description": "Test model", "training_info": {"samples": 100}}
                                )
                            
                            assert metadata.name == "test_model"
                            assert metadata.model_type == "regression"
                            assert metadata.description == "Test model"
                            assert metadata.version == 1
                            assert metadata.training_info["samples"] == 100

    @patch('src.persistence.get_config')
    def test_save_model_existing_model_no_overwrite(self, mock_get_config):
        """Test save_model method with existing model without overwrite."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        
        # Create model directory to simulate existing model
        model_path = storage._get_model_path("test_model")
        model_path.mkdir(parents=True)
        
        mock_model = Mock()
        
        with pytest.raises(ModelPersistenceError):
            storage.save_model("test_model", mock_model)

    @patch('src.persistence.get_config')
    def test_load_model_not_found(self, mock_get_config):
        """Test load_model method with non-existent model."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        
        with pytest.raises(ModelNotFoundError):
            storage.load_model("non_existent_model")

    @patch('src.persistence.get_config')
    def test_load_model_no_current_version(self, mock_get_config):
        """Test load_model method with no current version link."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        
        # Create model directory without current link
        model_path = storage._get_model_path("test_model")
        model_path.mkdir(parents=True)
        
        with pytest.raises(ModelPersistenceError):
            storage.load_model("test_model")

    @patch('src.persistence.get_config')
    def test_load_model_version_not_found(self, mock_get_config):
        """Test load_model method with non-existent version."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        
        # Create model directory
        model_path = storage._get_model_path("test_model")
        model_path.mkdir(parents=True)
        
        with pytest.raises(ModelNotFoundError):
            storage.load_model("test_model", version=5)

    @patch('src.persistence.get_config')
    def test_load_model_successful(self, mock_get_config):
        """Test successful load_model operation."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        
        # Setup model directory structure
        model_path = storage._get_model_path("test_model")
        version_path = storage._get_version_path("test_model", 1)
        version_path.mkdir(parents=True)
        
        # Create metadata file
        metadata_data = {
            "name": "test_model",
            "model_type": "regression",
            "version": 1,
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-01T00:00:00",
            "training_info": {},
            "performance_metrics": {},
            "feature_names": [],
            "storage_info": {"file_name": "model.json", "compression": False}
        }
        
        metadata_file = version_path / "info.json"
        metadata_file.write_text(json.dumps(metadata_data))
        
        # Create model file
        model_file = version_path / "model.json"
        model_file.write_text("test model content")
        
        # Create current symlink
        current_link = model_path / "current"
        current_link.symlink_to("versions/v1")
        
        with patch.object(storage, '_load_model_file') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            with patch.object(storage, '_calculate_checksum') as mock_checksum:
                mock_checksum.return_value = "test_checksum"
                
                model, metadata = storage.load_model("test_model", validate=False)
                
                assert model == mock_model
                assert metadata.name == "test_model"
                assert metadata.model_type == "regression"

    @patch('src.persistence.get_config')
    def test_list_models_empty(self, mock_get_config):
        """Test list_models method with empty storage."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        
        models = storage.list_models()
        
        assert isinstance(models, list)
        assert len(models) == 0

    @patch('src.persistence.get_config')
    def test_list_models_with_models(self, mock_get_config):
        """Test list_models method with existing models."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        
        # Create test model directories with metadata
        for i in range(3):
            model_name = f"model_{i}"
            model_path = storage._get_model_path(model_name)
            model_path.mkdir(parents=True)
            
            metadata_data = {
                "name": model_name,
                "model_type": "regression",
                "version": 1,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": f"2023-01-0{i+1}T00:00:00",
                "training_info": {},
                "performance_metrics": {},
                "feature_names": [],
                "storage_info": {}
            }
            
            metadata_file = model_path / "metadata.json"
            metadata_file.write_text(json.dumps(metadata_data))
        
        models = storage.list_models()
        
        assert len(models) == 3
        assert all(isinstance(m, ModelMetadata) for m in models)
        # Should be sorted by updated_at in descending order
        assert models[0].name == "model_2"

    @patch('src.persistence.get_config')
    def test_delete_model_not_found(self, mock_get_config):
        """Test delete_model method with non-existent model."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        
        with pytest.raises(ModelNotFoundError):
            storage.delete_model("non_existent_model")

    @patch('src.persistence.get_config')
    def test_delete_model_entire_model(self, mock_get_config):
        """Test delete_model method to delete entire model."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        
        # Create model directory
        model_path = storage._get_model_path("test_model")
        model_path.mkdir(parents=True)
        (model_path / "metadata.json").write_text("{}")
        
        result = storage.delete_model("test_model")
        
        assert result is True
        assert not model_path.exists()

    @patch('src.persistence.get_config')
    def test_delete_model_specific_version(self, mock_get_config):
        """Test delete_model method to delete specific version."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        
        # Create model directory with version
        model_path = storage._get_model_path("test_model")
        version_path = storage._get_version_path("test_model", 1)
        version_path.mkdir(parents=True)
        
        result = storage.delete_model("test_model", version=1)
        
        assert result is True
        assert not version_path.exists()

    @patch('src.persistence.get_config')
    def test_delete_model_version_not_found(self, mock_get_config):
        """Test delete_model method with non-existent version."""
        mock_get_config.return_value = self.mock_config
        
        storage = ModelStorage()
        
        # Create model directory
        model_path = storage._get_model_path("test_model")
        model_path.mkdir(parents=True)
        
        with pytest.raises(ModelNotFoundError):
            storage.delete_model("test_model", version=5)


class TestStorageManager:
    """Test StorageManager class."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "models"
        
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.storage.model_storage_path = str(self.storage_path)
        self.mock_config.storage.compression = False
        self.mock_config.storage.auto_save_models = True

    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.persistence.get_config')
    def test_storage_manager_initialization(self, mock_get_config):
        """Test StorageManager initialization."""
        mock_get_config.return_value = self.mock_config
        
        manager = StorageManager()
        
        assert manager.storage is not None
        assert isinstance(manager.storage, ModelStorage)

    @patch('src.persistence.get_config')
    def test_storage_manager_with_custom_storage(self, mock_get_config):
        """Test StorageManager with custom storage."""
        mock_get_config.return_value = self.mock_config
        
        custom_storage = Mock()
        manager = StorageManager(custom_storage)
        
        assert manager.storage == custom_storage

    @patch('src.persistence.get_config')
    def test_cleanup_old_versions(self, mock_get_config):
        """Test cleanup_old_versions method."""
        mock_get_config.return_value = self.mock_config
        
        manager = StorageManager()
        
        # Create mock model with multiple versions
        model_path = manager.storage._get_model_path("test_model")
        versions_path = model_path / "versions"
        
        # Create 5 versions
        for i in range(1, 6):
            version_path = versions_path / f"v{i}"
            version_path.mkdir(parents=True)
            (version_path / "model.json").write_text(f"model v{i}")
        
        # Mock list_models to return test model
        mock_metadata = Mock()
        mock_metadata.name = "test_model"
        manager.storage.list_models = Mock(return_value=[mock_metadata])
        
        # Keep only 3 versions
        deleted_count = manager.cleanup_old_versions(keep_versions=3)
        
        assert deleted_count == 2  # Should delete 2 oldest versions
        assert (versions_path / "v4").exists()
        assert (versions_path / "v5").exists()
        assert (versions_path / "v3").exists()
        assert not (versions_path / "v1").exists()
        assert not (versions_path / "v2").exists()

    @patch('src.persistence.get_config')
    def test_get_storage_stats(self, mock_get_config):
        """Test get_storage_stats method."""
        mock_get_config.return_value = self.mock_config
        
        manager = StorageManager()
        
        # Create mock models
        mock_models = []
        for i, (name, model_type) in enumerate([("model1", "regression"), ("model2", "classification"), ("model3", "regression")]):
            mock_model = Mock()
            mock_model.name = name
            mock_model.model_type = model_type
            mock_models.append(mock_model)
        
        manager.storage.list_models = Mock(return_value=mock_models)
        
        # Create test model directories
        for model in mock_models:
            model_path = manager.storage._get_model_path(model.name)
            model_path.mkdir(parents=True)
            (model_path / "test_file.txt").write_text("test content")
        
        stats = manager.get_storage_stats()
        
        assert isinstance(stats, dict)
        assert stats["total_models"] == 3
        assert stats["total_size_bytes"] > 0
        assert stats["total_size_mb"] > 0
        assert stats["model_types"]["regression"] == 2
        assert stats["model_types"]["classification"] == 1
        assert "storage_path" in stats


class TestConvenienceFunctions:
    """Test convenience functions."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "models"
        
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.storage.model_storage_path = str(self.storage_path)
        self.mock_config.storage.compression = False
        self.mock_config.storage.auto_save_models = True

    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.persistence.get_config')
    def test_save_model_function(self, mock_get_config):
        """Test save_model convenience function."""
        mock_get_config.return_value = self.mock_config
        
        mock_model = Mock()
        mock_model.save_model = Mock()
        
        with patch('builtins.hasattr', return_value=False):
            with patch('src.persistence.ModelStorage') as mock_storage_class:
                mock_storage = Mock()
                mock_metadata = Mock()
                mock_metadata.name = "test_model"
                mock_metadata.description = "Test"
                mock_storage.save_model.return_value = mock_metadata
                mock_storage_class.return_value = mock_storage
                
                metadata = save_model("test_model", mock_model, {"description": "Test"})
                
                assert metadata.name == "test_model"
                assert metadata.description == "Test"

    @patch('src.persistence.get_config')
    def test_load_model_function(self, mock_get_config):
        """Test load_model convenience function."""
        mock_get_config.return_value = self.mock_config
        
        with patch('src.persistence.ModelStorage') as mock_storage_class:
            mock_storage = Mock()
            mock_storage.load_model.return_value = ("model", "metadata")
            mock_storage_class.return_value = mock_storage
            
            model, metadata = load_model("test_model")
            
            assert model == "model"
            assert metadata == "metadata"
            mock_storage.load_model.assert_called_once_with("test_model", None, True)

    @patch('src.persistence.get_config')
    def test_list_models_function(self, mock_get_config):
        """Test list_models convenience function."""
        mock_get_config.return_value = self.mock_config
        
        with patch('src.persistence.ModelStorage') as mock_storage_class:
            mock_storage = Mock()
            mock_storage.list_models.return_value = ["model1", "model2"]
            mock_storage_class.return_value = mock_storage
            
            models = list_models()
            
            assert models == ["model1", "model2"]
            mock_storage.list_models.assert_called_once()

    @patch('src.persistence.get_config')
    def test_delete_model_function(self, mock_get_config):
        """Test delete_model convenience function."""
        mock_get_config.return_value = self.mock_config
        
        with patch('src.persistence.ModelStorage') as mock_storage_class:
            mock_storage = Mock()
            mock_storage.delete_model.return_value = True
            mock_storage_class.return_value = mock_storage
            
            result = delete_model("test_model")
            
            assert result is True
            mock_storage.delete_model.assert_called_once_with("test_model", None)

    @patch('src.persistence.get_config')
    def test_cleanup_storage_function(self, mock_get_config):
        """Test cleanup_storage convenience function."""
        mock_get_config.return_value = self.mock_config
        
        with patch('src.persistence.StorageManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.cleanup_old_versions.return_value = 5
            mock_manager_class.return_value = mock_manager
            
            result = cleanup_storage(keep_versions=2)
            
            assert result == 5
            mock_manager.cleanup_old_versions.assert_called_once_with(2)


def mock_open_multiple_files(files_dict):
    """Helper to mock multiple file opens."""
    def mock_open_func(*args, **kwargs):
        filename = str(args[0])  # Convert Path objects to string
        if filename in files_dict:
            return Mock()
        else:
            # Create a mock file object that behaves like a real file
            mock_file = Mock()
            mock_file.read.return_value = "test content"
            mock_file.write.return_value = None
            mock_file.__enter__.return_value = mock_file
            mock_file.__exit__.return_value = None
            return mock_file
    return mock_open_func 