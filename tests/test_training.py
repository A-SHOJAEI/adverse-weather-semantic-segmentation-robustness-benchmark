"""Tests for training pipeline and components."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil

from adverse_weather_semantic_segmentation_robustness_benchmark.training.trainer import (
    AdverseWeatherTrainer, EarlyStopping
)
from adverse_weather_semantic_segmentation_robustness_benchmark.models.model import (
    SegFormerModel, FogDensityAwareLoss
)
from adverse_weather_semantic_segmentation_robustness_benchmark.data.loader import (
    CityscapesKITTIDataset, create_dataloader
)
from adverse_weather_semantic_segmentation_robustness_benchmark.evaluation.metrics import (
    RobustnessMetrics
)


class TestEarlyStopping:
    """Test EarlyStopping utility."""

    def test_early_stopping_initialization(self):
        """Test early stopping initialization."""
        early_stopping = EarlyStopping(
            patience=5,
            min_delta=0.001,
            restore_best_weights=True
        )

        assert early_stopping.patience == 5
        assert early_stopping.min_delta == 0.001
        assert early_stopping.restore_best_weights is True
        assert early_stopping.best_loss == float('inf')
        assert early_stopping.counter == 0
        assert early_stopping.early_stop is False

    def test_early_stopping_improvement(self):
        """Test early stopping when validation loss improves."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)

        # Create dummy model
        model = nn.Linear(10, 1)

        # First call with high loss
        should_stop = early_stopping(1.0, model)
        assert not should_stop
        assert early_stopping.counter == 0

        # Second call with lower loss (improvement)
        should_stop = early_stopping(0.5, model)
        assert not should_stop
        assert early_stopping.counter == 0

        # Third call with much lower loss (improvement)
        should_stop = early_stopping(0.1, model)
        assert not should_stop
        assert early_stopping.counter == 0

    def test_early_stopping_no_improvement(self):
        """Test early stopping when validation loss doesn't improve."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01)

        # Create dummy model
        model = nn.Linear(10, 1)

        # First call
        early_stopping(1.0, model)

        # Second call with worse loss
        should_stop = early_stopping(1.1, model)
        assert not should_stop
        assert early_stopping.counter == 1

        # Third call with worse loss - should trigger early stopping
        should_stop = early_stopping(1.2, model)
        assert should_stop
        assert early_stopping.early_stop is True

    def test_early_stopping_weight_restoration(self):
        """Test weight restoration functionality."""
        early_stopping = EarlyStopping(patience=1, restore_best_weights=True)

        # Create model and save initial weights
        model = nn.Linear(10, 1)
        initial_weight = model.weight.clone()

        # First call (this will be the best)
        early_stopping(0.5, model)

        # Modify model weights
        with torch.no_grad():
            model.weight.fill_(999.0)

        # Second call with worse loss - should restore weights
        early_stopping(1.0, model)

        # Weights should be restored to best (initial) values
        assert torch.allclose(model.weight, initial_weight)


class TestAdverseWeatherTrainer:
    """Test AdverseWeatherTrainer class."""

    @pytest.fixture
    def mock_data_loaders(self, test_config, synthetic_dataset_dir):
        """Create mock data loaders for testing."""
        # Create small datasets for testing
        train_dataset = CityscapesKITTIDataset(
            data_root=str(synthetic_dataset_dir),
            split='train',
            image_size=(64, 128),
            weather_conditions=['clean', 'fog'],
            dataset_type='synthetic'
        )

        val_dataset = CityscapesKITTIDataset(
            data_root=str(synthetic_dataset_dir),
            split='val',
            image_size=(64, 128),
            weather_conditions=['clean', 'fog'],
            dataset_type='synthetic'
        )

        train_loader = create_dataloader(train_dataset, batch_size=2, num_workers=0)
        val_loader = create_dataloader(val_dataset, batch_size=2, num_workers=0)

        return train_loader, val_loader

    def test_trainer_initialization(self, test_config, mock_data_loaders, device, temp_dir):
        """Test trainer initialization."""
        train_loader, val_loader = mock_data_loaders

        model = SegFormerModel(num_classes=5, include_depth=False, pretrained=False)

        trainer = AdverseWeatherTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=test_config.to_dict(),
            device=device,
            checkpoint_dir=str(temp_dir / 'checkpoints'),
            log_dir=str(temp_dir / 'logs')
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.loss_fn is not None
        assert trainer.metrics is not None
        assert trainer.device == device

    def test_trainer_optimizer_setup(self, test_config, mock_data_loaders, device, temp_dir):
        """Test optimizer setup in trainer."""
        train_loader, val_loader = mock_data_loaders
        model = SegFormerModel(num_classes=5, include_depth=False, pretrained=False)

        # Test different optimizers
        for optimizer_type in ['adamw', 'sgd', 'adam']:
            config = test_config.to_dict()
            config['optimizer']['type'] = optimizer_type

            trainer = AdverseWeatherTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=device,
                checkpoint_dir=str(temp_dir / 'checkpoints'),
                log_dir=str(temp_dir / 'logs')
            )

            assert trainer.optimizer is not None

    def test_trainer_loss_function_setup(self, test_config, mock_data_loaders, device, temp_dir):
        """Test loss function setup in trainer."""
        train_loader, val_loader = mock_data_loaders
        model = SegFormerModel(num_classes=5, include_depth=False, pretrained=False)

        # Test different loss functions
        for loss_type in ['cross_entropy', 'fog_density_aware']:
            config = test_config.to_dict()
            config['loss']['type'] = loss_type

            trainer = AdverseWeatherTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=device,
                checkpoint_dir=str(temp_dir / 'checkpoints'),
                log_dir=str(temp_dir / 'logs')
            )

            assert trainer.loss_fn is not None

    def test_trainer_single_epoch(self, test_config, mock_data_loaders, device, temp_dir):
        """Test single epoch training."""
        train_loader, val_loader = mock_data_loaders
        model = SegFormerModel(num_classes=5, include_depth=False, pretrained=False)

        # Disable MLflow for testing
        config = test_config.to_dict()
        config['mlflow']['enabled'] = False

        trainer = AdverseWeatherTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_dir=str(temp_dir / 'checkpoints'),
            log_dir=str(temp_dir / 'logs')
        )

        # Run single training epoch
        train_metrics = trainer.train_epoch()

        assert isinstance(train_metrics, dict)
        assert 'train_loss' in train_metrics
        assert 'train_seg_loss' in train_metrics
        assert train_metrics['train_loss'] >= 0
        assert train_metrics['train_samples'] > 0

    def test_trainer_single_validation(self, test_config, mock_data_loaders, device, temp_dir):
        """Test single validation epoch."""
        train_loader, val_loader = mock_data_loaders
        model = SegFormerModel(num_classes=5, include_depth=False, pretrained=False)

        # Disable MLflow for testing
        config = test_config.to_dict()
        config['mlflow']['enabled'] = False

        trainer = AdverseWeatherTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_dir=str(temp_dir / 'checkpoints'),
            log_dir=str(temp_dir / 'logs')
        )

        # Run single validation epoch
        val_metrics = trainer.validate_epoch()

        assert isinstance(val_metrics, dict)
        assert 'val_loss' in val_metrics
        assert 'val_miou' in val_metrics
        assert val_metrics['val_loss'] >= 0
        assert val_metrics['val_miou'] >= 0
        assert val_metrics['val_samples'] > 0

    def test_trainer_checkpoint_saving(self, test_config, mock_data_loaders, device, temp_dir):
        """Test checkpoint saving functionality."""
        train_loader, val_loader = mock_data_loaders
        model = SegFormerModel(num_classes=5, include_depth=False, pretrained=False)

        checkpoint_dir = temp_dir / 'checkpoints'
        checkpoint_dir.mkdir()

        # Disable MLflow for testing
        config = test_config.to_dict()
        config['mlflow']['enabled'] = False

        trainer = AdverseWeatherTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_dir=str(checkpoint_dir),
            log_dir=str(temp_dir / 'logs')
        )

        # Save checkpoint
        test_metrics = {'val_loss': 0.5, 'val_miou': 0.7}
        trainer.save_checkpoint(
            epoch=0,
            metrics=test_metrics,
            is_best=True
        )

        # Check that checkpoint files exist
        assert (checkpoint_dir / 'latest.pth').exists()
        assert (checkpoint_dir / 'best.pth').exists()

        # Load checkpoint and verify
        checkpoint = torch.load(checkpoint_dir / 'best.pth', map_location='cpu')
        assert checkpoint['epoch'] == 0
        assert checkpoint['metrics']['val_loss'] == 0.5
        assert 'model_state_dict' in checkpoint

    def test_trainer_checkpoint_loading(self, test_config, mock_data_loaders, device, temp_dir):
        """Test checkpoint loading functionality."""
        train_loader, val_loader = mock_data_loaders
        model = SegFormerModel(num_classes=5, include_depth=False, pretrained=False)

        checkpoint_dir = temp_dir / 'checkpoints'
        checkpoint_dir.mkdir()

        # Disable MLflow for testing
        config = test_config.to_dict()
        config['mlflow']['enabled'] = False

        trainer = AdverseWeatherTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_dir=str(checkpoint_dir),
            log_dir=str(temp_dir / 'logs')
        )

        # Save a checkpoint first
        test_metrics = {'val_loss': 0.5}
        trainer.save_checkpoint(epoch=5, metrics=test_metrics)

        # Load the checkpoint
        trainer.load_checkpoint(str(checkpoint_dir / 'latest.pth'))

        # Verify that epoch was loaded
        assert trainer.current_epoch == 5

    def test_trainer_full_training_loop_short(self, test_config, mock_data_loaders, device, temp_dir):
        """Test full training loop with very few epochs."""
        train_loader, val_loader = mock_data_loaders
        model = SegFormerModel(num_classes=5, include_depth=False, pretrained=False)

        # Configure for very short training
        config = test_config.to_dict()
        config['epochs'] = 2  # Very short training
        config['mlflow']['enabled'] = False
        config['scheduler']['enabled'] = False

        trainer = AdverseWeatherTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_dir=str(temp_dir / 'checkpoints'),
            log_dir=str(temp_dir / 'logs')
        )

        # Run training
        results = trainer.train()

        # Check results
        assert isinstance(results, dict)
        assert 'history' in results
        assert 'best_val_miou' in results
        assert 'total_epochs' in results
        assert results['total_epochs'] <= 2

    def test_fog_density_estimation_in_trainer(self, test_config, mock_data_loaders, device, temp_dir):
        """Test fog density estimation in trainer."""
        train_loader, val_loader = mock_data_loaders
        model = SegFormerModel(num_classes=5, include_depth=True, pretrained=False)

        # Use fog-density-aware loss
        config = test_config.to_dict()
        config['loss']['type'] = 'fog_density_aware'
        config['mlflow']['enabled'] = False

        trainer = AdverseWeatherTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_dir=str(temp_dir / 'checkpoints'),
            log_dir=str(temp_dir / 'logs')
        )

        # Test fog density estimation
        test_batch = {
            'weather_condition': ['fog', 'clean'],
            'image': torch.randn(2, 3, 64, 128)
        }

        fog_density = trainer._estimate_fog_density(test_batch)

        assert fog_density is not None
        assert fog_density.shape == (2, 64, 128)
        assert torch.all(fog_density >= 0) and torch.all(fog_density <= 1)


class TestTrainingIntegration:
    """Integration tests for training components."""

    def test_training_with_fog_density_aware_loss(self, test_config, synthetic_dataset_dir, device):
        """Test training with fog-density-aware loss."""
        # Create small dataset
        train_dataset = CityscapesKITTIDataset(
            data_root=str(synthetic_dataset_dir),
            split='train',
            image_size=(32, 64),  # Very small for fast testing
            weather_conditions=['clean', 'fog'],
            dataset_type='synthetic'
        )

        train_loader = create_dataloader(train_dataset, batch_size=1, num_workers=0)

        # Create model with depth estimation
        model = SegFormerModel(num_classes=5, include_depth=True, pretrained=False)

        # Test fog-density-aware loss
        loss_fn = FogDensityAwareLoss()

        # Get a batch
        batch = next(iter(train_loader))
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        outputs = model(images)

        # Prepare targets
        targets = {'label': labels}
        if 'depth' in batch:
            targets['depth'] = batch['depth'].to(device)

        # Compute loss
        loss_dict = loss_fn(outputs, targets)

        assert 'total_loss' in loss_dict
        assert loss_dict['total_loss'].item() >= 0

    def test_metrics_computation_during_training(self, test_config, mock_data_loaders, device, temp_dir):
        """Test metrics computation during training."""
        train_loader, val_loader = mock_data_loaders
        model = SegFormerModel(num_classes=5, include_depth=False, pretrained=False)

        # Initialize metrics
        metrics = RobustnessMetrics(num_classes=5)

        # Test metrics computation on a batch
        batch = next(iter(val_loader))
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = model(images)
            predictions = outputs['segmentation'].argmax(dim=1)

        # Compute mIoU
        miou = metrics.compute_miou(predictions, labels)

        assert isinstance(miou, float)
        assert miou >= 0 and miou <= 1

    def test_early_stopping_integration(self, test_config, mock_data_loaders, device, temp_dir):
        """Test early stopping integration with trainer."""
        train_loader, val_loader = mock_data_loaders
        model = SegFormerModel(num_classes=5, include_depth=False, pretrained=False)

        # Configure early stopping with very low patience
        config = test_config.to_dict()
        config['early_stopping']['patience'] = 1
        config['epochs'] = 10  # More than patience to test early stopping
        config['mlflow']['enabled'] = False

        trainer = AdverseWeatherTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_dir=str(temp_dir / 'checkpoints'),
            log_dir=str(temp_dir / 'logs')
        )

        # Run training (should stop early)
        results = trainer.train()

        # Training should have stopped before reaching 10 epochs
        assert results['total_epochs'] < 10