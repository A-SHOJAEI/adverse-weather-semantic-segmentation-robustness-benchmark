"""Pytest configuration and fixtures for test suite."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Import modules for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adverse_weather_semantic_segmentation_robustness_benchmark.utils.config import Config
from adverse_weather_semantic_segmentation_robustness_benchmark.models.model import (
    SegFormerModel, DeepLabV3PlusModel, EnsembleModel
)


@pytest.fixture
def device():
    """Provide device for testing."""
    return torch.device('cpu')  # Use CPU for tests


@pytest.fixture
def test_config():
    """Provide test configuration."""
    config_dict = {
        'model': {
            'type': 'ensemble',
            'num_classes': 5,  # Smaller for testing
            'include_depth': True,
            'ensemble_strategy': 'weighted_average',
            'temperature_scaling': True
        },
        'data': {
            'dataset_type': 'synthetic',
            'data_root': 'test_data',
            'image_size': [256, 512],  # Smaller for testing
            'weather_conditions': ['clean', 'fog', 'rain'],
            'apply_augmentation': True,
            'include_depth': True
        },
        'training': {
            'batch_size': 2,
            'epochs': 3,  # Few epochs for testing
            'num_workers': 0,  # No multiprocessing in tests
            'pin_memory': False,
            'grad_clip': 1.0
        },
        'optimizer': {
            'type': 'adamw',
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'betas': [0.9, 0.999]
        },
        'scheduler': {
            'enabled': False  # Disable for testing
        },
        'loss': {
            'type': 'cross_entropy',  # Simple loss for testing
            'base_loss': 'cross_entropy',
            'depth_weight': 0.5,
            'fog_sensitivity': 2.0,
            'depth_loss_weight': 0.1
        },
        'early_stopping': {
            'patience': 2,
            'min_delta': 0.001,
            'restore_best_weights': True
        },
        'mlflow': {
            'enabled': False  # Disable MLflow in tests
        },
        'evaluation': {
            'num_bins': 5,  # Fewer bins for testing
            'weather_conditions': ['clean', 'fog', 'rain']
        },
        'logging': {
            'level': 'WARNING'  # Reduce log noise in tests
        },
        'paths': {
            'checkpoints': 'test_checkpoints',
            'logs': 'test_logs',
            'results': 'test_results'
        },
        'device': 'cpu',
        'seed': 42
    }

    return Config(config_dict)


@pytest.fixture
def sample_batch():
    """Provide sample batch data for testing."""
    batch_size = 2
    height, width = 256, 512
    num_classes = 5

    batch = {
        'image': torch.randn(batch_size, 3, height, width),
        'label': torch.randint(0, num_classes, (batch_size, height, width)),
        'depth': torch.rand(batch_size, height, width),
        'weather_condition': ['clean', 'fog']
    }

    return batch


@pytest.fixture
def small_sample_batch():
    """Provide small sample batch for quick testing."""
    batch_size = 1
    height, width = 64, 128
    num_classes = 5

    batch = {
        'image': torch.randn(batch_size, 3, height, width),
        'label': torch.randint(0, num_classes, (batch_size, height, width)),
        'depth': torch.rand(batch_size, height, width),
        'weather_condition': ['clean']
    }

    return batch


@pytest.fixture
def segformer_model(device):
    """Provide SegFormer model for testing."""
    model = SegFormerModel(
        num_classes=5,
        include_depth=True,
        pretrained=False
    )
    return model.to(device)


@pytest.fixture
def deeplabv3plus_model(device):
    """Provide DeepLabV3+ model for testing."""
    model = DeepLabV3PlusModel(
        num_classes=5,
        include_depth=True,
        pretrained=False
    )
    return model.to(device)


@pytest.fixture
def ensemble_model(device):
    """Provide ensemble model for testing."""
    model = EnsembleModel(
        num_classes=5,
        include_depth=True,
        ensemble_strategy='weighted_average',
        temperature_scaling=True
    )
    return model.to(device)


@pytest.fixture
def temp_dir():
    """Provide temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def synthetic_dataset_dir(temp_dir):
    """Create synthetic dataset directory structure."""
    data_dir = temp_dir / "data"
    data_dir.mkdir()

    # Create some dummy files
    (data_dir / "train").mkdir()
    (data_dir / "val").mkdir()
    (data_dir / "test").mkdir()

    return data_dir


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def sample_predictions():
    """Provide sample predictions for metric testing."""
    batch_size = 2
    num_classes = 5
    height, width = 64, 128

    # Create logits and convert to predictions
    logits = torch.randn(batch_size, num_classes, height, width)
    predictions = logits.argmax(dim=1)
    targets = torch.randint(0, num_classes, (batch_size, height, width))

    return {
        'logits': logits,
        'predictions': predictions,
        'targets': targets
    }


@pytest.fixture
def weather_predictions():
    """Provide weather-specific predictions for robustness testing."""
    batch_size = 2
    num_classes = 5
    height, width = 64, 128

    weather_data = {}

    for weather in ['clean', 'fog', 'rain']:
        logits = torch.randn(batch_size, num_classes, height, width)
        predictions = logits.argmax(dim=1)
        targets = torch.randint(0, num_classes, (batch_size, height, width))

        weather_data[weather] = {
            'logits': logits,
            'predictions': predictions,
            'targets': targets
        }

    return weather_data