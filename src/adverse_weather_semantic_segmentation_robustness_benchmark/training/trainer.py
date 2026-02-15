"""Training pipeline for adverse weather semantic segmentation models."""

import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# MLflow imports with error handling
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

from ..models.model import EnsembleModel, FogDensityAwareLoss
from ..evaluation.metrics import RobustnessMetrics
from ..data.preprocessing import WeatherDegradationTransforms

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.

    Monitors validation loss and stops training when no improvement
    is observed for a specified number of epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        restore_best_weights: bool = True
    ) -> None:
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should be stopped.

        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights

        Returns:
            Whether to stop training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)

        return self.early_stop


class AdverseWeatherTrainer:
    """
    Comprehensive trainer for adverse weather semantic segmentation models.

    Implements domain adaptation training with weather augmentation,
    ensemble learning, and comprehensive evaluation metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs"
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to use for training
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize optimizer
        self.optimizer = self._setup_optimizer()

        # Initialize scheduler
        self.scheduler = self._setup_scheduler()

        # Initialize loss function
        self.loss_fn = self._setup_loss_function()

        # Initialize metrics
        self.metrics = RobustnessMetrics(num_classes=config.get('num_classes', 19))

        # Weather transforms for domain adaptation
        self.weather_transforms = WeatherDegradationTransforms()

        # Early stopping
        early_stop_config = config.get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=early_stop_config.get('patience', 10),
            min_delta=early_stop_config.get('min_delta', 0.001),
            restore_best_weights=early_stop_config.get('restore_best_weights', True)
        )

        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_miou = 0.0

        # Initialize MLflow
        self._setup_mlflow()

        logger.info(f"Initialized AdverseWeatherTrainer with {type(model).__name__}")

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on configuration."""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adamw')
        learning_rate = optimizer_config.get('learning_rate', 0.001)
        weight_decay = optimizer_config.get('weight_decay', 0.01)

        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

        return optimizer

    def _setup_scheduler(self) -> Optional[optim.lr_scheduler.LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        if not scheduler_config.get('enabled', False):
            return None

        scheduler_type = scheduler_config.get('type', 'cosine')

        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config.get('patience', 5),
                factor=scheduler_config.get('factor', 0.5)
            )
        else:
            return None

        return scheduler

    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function based on configuration."""
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'fog_density_aware')

        if loss_type == 'fog_density_aware':
            loss_fn = FogDensityAwareLoss(
                base_loss=loss_config.get('base_loss', 'cross_entropy'),
                depth_weight=loss_config.get('depth_weight', 0.5),
                fog_sensitivity=loss_config.get('fog_sensitivity', 2.0),
                depth_loss_weight=loss_config.get('depth_loss_weight', 0.1)
            )
        elif loss_type == 'cross_entropy':
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()

        return loss_fn

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Skipping MLflow setup.")
            return

        try:
            mlflow_config = self.config.get('mlflow', {})
            if mlflow_config.get('enabled', True):
                experiment_name = mlflow_config.get('experiment_name', 'adverse_weather_segmentation')
                mlflow.set_experiment(experiment_name)

                # Start MLflow run
                mlflow.start_run(run_name=mlflow_config.get('run_name', None))

                # Log hyperparameters
                mlflow.log_params({
                    'model_type': type(self.model).__name__,
                    'optimizer': self.config.get('optimizer', {}).get('type', 'adamw'),
                    'learning_rate': self.config.get('optimizer', {}).get('learning_rate', 0.001),
                    'batch_size': self.config.get('batch_size', 8),
                    'epochs': self.config.get('epochs', 100),
                    'num_classes': self.config.get('num_classes', 19)
                })

                logger.info("MLflow tracking initialized")
        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_metrics = {
            'train_loss': 0.0,
            'train_seg_loss': 0.0,
            'train_depth_loss': 0.0,
            'train_samples': 0
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Get depth if available
            depths = batch.get('depth')
            if depths is not None:
                depths = depths.to(self.device)

            # Forward pass
            outputs = self.model(images)

            # Prepare targets for loss computation
            targets = {'label': labels}
            if depths is not None:
                targets['depth'] = depths

            # Compute fog density for fog-density-aware loss
            fog_density = None
            if isinstance(self.loss_fn, FogDensityAwareLoss):
                # Estimate fog density from weather condition or depth
                fog_density = self._estimate_fog_density(batch)
                if fog_density is not None:
                    fog_density = fog_density.to(self.device)

            # Compute loss
            if isinstance(self.loss_fn, FogDensityAwareLoss):
                loss_dict = self.loss_fn(outputs, targets, fog_density)
                loss = loss_dict['total_loss']
                seg_loss = loss_dict['segmentation_loss']
                depth_loss = loss_dict['depth_loss']
            else:
                seg_loss = self.loss_fn(outputs['segmentation'], labels)
                depth_loss = 0.0
                loss = seg_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_clip = self.config.get('grad_clip', 1.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()

            # Update metrics
            batch_size = images.size(0)
            epoch_metrics['train_loss'] += loss.item() * batch_size
            epoch_metrics['train_seg_loss'] += seg_loss.item() * batch_size
            if isinstance(depth_loss, torch.Tensor):
                epoch_metrics['train_depth_loss'] += depth_loss.item() * batch_size
            else:
                epoch_metrics['train_depth_loss'] += depth_loss * batch_size
            epoch_metrics['train_samples'] += batch_size

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'seg_loss': f"{seg_loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

            # Log to tensorboard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/SegLoss', seg_loss.item(), self.global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        # Average metrics over epoch
        for key in epoch_metrics:
            if key != 'train_samples':
                epoch_metrics[key] /= epoch_metrics['train_samples']

        return epoch_metrics

    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_metrics = {
            'val_loss': 0.0,
            'val_seg_loss': 0.0,
            'val_depth_loss': 0.0,
            'val_samples': 0
        }

        # Initialize metrics accumulators
        all_predictions = []
        all_targets = []
        weather_predictions = {'clean': [], 'fog': [], 'rain': [], 'snow': [], 'night': []}
        weather_targets = {'clean': [], 'fog': [], 'rain': [], 'snow': [], 'night': []}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                weather_conditions = batch.get('weather_condition', ['clean'] * images.size(0))

                # Get depth if available
                depths = batch.get('depth')
                if depths is not None:
                    depths = depths.to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Prepare targets for loss computation
                targets = {'label': labels}
                if depths is not None:
                    targets['depth'] = depths

                # Compute fog density for fog-density-aware loss
                fog_density = None
                if isinstance(self.loss_fn, FogDensityAwareLoss):
                    fog_density = self._estimate_fog_density(batch)
                    if fog_density is not None:
                        fog_density = fog_density.to(self.device)

                # Compute loss
                if isinstance(self.loss_fn, FogDensityAwareLoss):
                    loss_dict = self.loss_fn(outputs, targets, fog_density)
                    loss = loss_dict['total_loss']
                    seg_loss = loss_dict['segmentation_loss']
                    depth_loss = loss_dict['depth_loss']
                else:
                    seg_loss = self.loss_fn(outputs['segmentation'], labels)
                    depth_loss = 0.0
                    loss = seg_loss

                # Update metrics
                batch_size = images.size(0)
                val_metrics['val_loss'] += loss.item() * batch_size
                val_metrics['val_seg_loss'] += seg_loss.item() * batch_size
                if isinstance(depth_loss, torch.Tensor):
                    val_metrics['val_depth_loss'] += depth_loss.item() * batch_size
                else:
                    val_metrics['val_depth_loss'] += depth_loss * batch_size
                val_metrics['val_samples'] += batch_size

                # Collect predictions and targets for metrics computation
                predictions = outputs['segmentation'].argmax(dim=1)
                all_predictions.append(predictions.cpu())
                all_targets.append(labels.cpu())

                # Group by weather condition
                for i, weather in enumerate(weather_conditions):
                    if weather in weather_predictions:
                        weather_predictions[weather].append(predictions[i:i+1].cpu())
                        weather_targets[weather].append(labels[i:i+1].cpu())

        # Average metrics over epoch
        for key in val_metrics:
            if key != 'val_samples':
                val_metrics[key] /= val_metrics['val_samples']

        # Compute robustness metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Overall mIoU
        overall_miou = self.metrics.compute_miou(all_predictions, all_targets)
        val_metrics['val_miou'] = overall_miou

        # Weather-specific mIoU
        for weather, preds_list in weather_predictions.items():
            if preds_list:
                weather_preds = torch.cat(preds_list, dim=0)
                weather_tgts = torch.cat(weather_targets[weather], dim=0)
                weather_miou = self.metrics.compute_miou(weather_preds, weather_tgts)
                val_metrics[f'val_miou_{weather}'] = weather_miou

        return val_metrics

    def _estimate_fog_density(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Estimate fog density from batch information.

        Args:
            batch: Input batch

        Returns:
            Fog density map or None
        """
        weather_conditions = batch.get('weather_condition', [])
        if not weather_conditions:
            return None

        # Simple fog density estimation based on weather condition
        batch_size = len(weather_conditions)
        h, w = batch['image'].shape[2:]

        fog_density = torch.zeros(batch_size, h, w)

        for i, weather in enumerate(weather_conditions):
            if weather == 'fog':
                # High fog density for foggy images
                fog_density[i] = torch.rand(h, w) * 0.5 + 0.5
            elif weather in ['rain', 'snow']:
                # Moderate fog density for other adverse conditions
                fog_density[i] = torch.rand(h, w) * 0.3 + 0.2
            else:
                # Low fog density for clean conditions
                fog_density[i] = torch.rand(h, w) * 0.1

        return fog_density

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Training history and final metrics
        """
        num_epochs = self.config.get('epochs', 100)
        history = {'train': [], 'val': []}

        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Train epoch
            train_metrics = self.train_epoch()
            history['train'].append(train_metrics)

            # Validate epoch
            val_metrics = self.validate_epoch()
            history['val'].append(val_metrics)

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            # Logging
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val mIoU: {val_metrics['val_miou']:.4f}, "
                f"Time: {epoch_time:.1f}s"
            )

            # Log to tensorboard
            self.writer.add_scalar('Epoch/TrainLoss', train_metrics['train_loss'], epoch)
            self.writer.add_scalar('Epoch/ValLoss', val_metrics['val_loss'], epoch)
            self.writer.add_scalar('Epoch/ValMIoU', val_metrics['val_miou'], epoch)

            # Log to MLflow
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metrics({
                        'train_loss': train_metrics['train_loss'],
                        'val_loss': val_metrics['val_loss'],
                        'val_miou': val_metrics['val_miou']
                    }, step=epoch)
                except Exception as e:
                    logger.warning(f"Failed to log to MLflow: {e}")

            # Save checkpoint
            is_best = val_metrics['val_miou'] > self.best_val_miou
            if is_best:
                self.best_val_miou = val_metrics['val_miou']
                self.best_val_loss = val_metrics['val_loss']

            self.save_checkpoint(
                epoch=epoch,
                metrics=val_metrics,
                is_best=is_best
            )

            # Early stopping check
            if self.early_stopping(val_metrics['val_loss'], self.model):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Close tensorboard writer
        self.writer.close()

        # End MLflow run
        if MLFLOW_AVAILABLE:
            try:
                mlflow.end_run()
            except Exception:
                pass

        logger.info("Training completed")

        return {
            'history': history,
            'best_val_miou': self.best_val_miou,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1
        }

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with mIoU: {metrics['val_miou']:.4f}")

        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            periodic_path = self.checkpoint_dir / f"epoch_{epoch + 1}.pth"
            torch.save(checkpoint, periodic_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch + 1}")

    def resume_training(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Training results
        """
        self.load_checkpoint(checkpoint_path)
        return self.train()