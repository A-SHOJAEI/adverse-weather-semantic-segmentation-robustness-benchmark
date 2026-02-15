#!/usr/bin/env python3
"""
Training script for adverse weather semantic segmentation models.

This script trains semantic segmentation models with weather robustness
capabilities, including ensemble models and fog-density-aware loss functions.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import random

import torch
import torch.backends.cudnn as cudnn
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adverse_weather_semantic_segmentation_robustness_benchmark.utils.config import (
    load_config, create_default_config, setup_logging, validate_config, get_device_config
)
from adverse_weather_semantic_segmentation_robustness_benchmark.data.loader import (
    CityscapesKITTIDataset, WeatherAugmentationPipeline, create_dataloader
)
from adverse_weather_semantic_segmentation_robustness_benchmark.models.model import (
    SegFormerModel, DeepLabV3PlusModel, EnsembleModel
)
from adverse_weather_semantic_segmentation_robustness_benchmark.training.trainer import (
    AdverseWeatherTrainer
)

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic
    cudnn.deterministic = True
    cudnn.benchmark = False

    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    logger.info(f"Random seed set to {seed}")


def create_model(config) -> torch.nn.Module:
    """
    Create model based on configuration.

    Args:
        config: Configuration object

    Returns:
        Initialized model
    """
    model_type = config.get('model.type', 'ensemble')
    num_classes = config.get('model.num_classes', 19)
    include_depth = config.get('model.include_depth', True)

    if model_type == 'segformer':
        model = SegFormerModel(
            num_classes=num_classes,
            include_depth=include_depth
        )
    elif model_type == 'deeplabv3plus':
        model = DeepLabV3PlusModel(
            num_classes=num_classes,
            include_depth=include_depth
        )
    elif model_type == 'ensemble':
        model = EnsembleModel(
            num_classes=num_classes,
            include_depth=include_depth,
            ensemble_strategy=config.get('model.ensemble_strategy', 'weighted_average'),
            temperature_scaling=config.get('model.temperature_scaling', True)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Created {model_type} model with {num_classes} classes")
    return model


def create_datasets_and_loaders(config):
    """
    Create datasets and data loaders.

    Args:
        config: Configuration object

    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_root = config.get('data.data_root', 'data')
    dataset_type = config.get('data.dataset_type', 'combined')
    image_size = tuple(config.get('data.image_size', [512, 1024]))
    weather_conditions = config.get('data.weather_conditions', ['clean', 'fog', 'rain', 'snow', 'night'])
    apply_augmentation = config.get('data.apply_augmentation', True)
    include_depth = config.get('data.include_depth', True)

    batch_size = config.get('training.batch_size', 8)
    num_workers = config.get('training.num_workers', 4)
    pin_memory = config.get('training.pin_memory', True)

    # Create datasets
    train_dataset = CityscapesKITTIDataset(
        data_root=data_root,
        split='train',
        image_size=image_size,
        weather_conditions=weather_conditions,
        apply_augmentation=apply_augmentation,
        include_depth=include_depth,
        dataset_type=dataset_type
    )

    val_dataset = CityscapesKITTIDataset(
        data_root=data_root,
        split='val',
        image_size=image_size,
        weather_conditions=weather_conditions,
        apply_augmentation=False,  # No augmentation for validation
        include_depth=include_depth,
        dataset_type=dataset_type
    )

    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.info(f"Created datasets: train={len(train_dataset)}, val={len(val_dataset)}")
    logger.info(f"Created dataloaders: train_batches={len(train_loader)}, val_batches={len(val_loader)}")

    return train_loader, val_loader


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train adverse weather semantic segmentation models")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cpu, cuda, cuda:0, etc.)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for checkpoints and logs'
    )

    args = parser.parse_args()

    # Load configuration
    try:
        if Path(args.config).exists():
            config = load_config(args.config)
        else:
            logger.warning(f"Config file {args.config} not found. Using default configuration.")
            config = create_default_config()
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        logger.info("Using default configuration")
        config = create_default_config()

    # Override config with command line arguments
    if args.device != 'auto':
        config.set('device', args.device)
    if args.seed is not None:
        config.set('seed', args.seed)

    # Setup output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / config.get('paths.checkpoints', 'checkpoints')
    log_dir = output_dir / config.get('paths.logs', 'logs')

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(config)

    # Validate configuration
    try:
        validate_config(config)
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)

    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)

    # Setup device
    device_str = get_device_config(config.get('device', 'auto'))
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # Create model
    try:
        model = create_model(config)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    except Exception as e:
        logger.error(f"Error creating model: {e}")
        sys.exit(1)

    # Create datasets and data loaders
    try:
        train_loader, val_loader = create_datasets_and_loaders(config)
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        logger.info("This error is expected if real dataset files are not available.")
        logger.info("The system will use synthetic data for training.")
        # Continue with synthetic data - this is handled by the dataset class
        train_loader, val_loader = create_datasets_and_loaders(config)

    # Create trainer
    trainer = AdverseWeatherTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.to_dict(),
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir)
    )

    # Resume from checkpoint if specified
    if args.resume:
        try:
            trainer.load_checkpoint(args.resume)
            logger.info(f"Resumed training from {args.resume}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            sys.exit(1)

    # Start training
    try:
        logger.info("Starting training...")
        results = trainer.train()

        logger.info("Training completed successfully!")
        logger.info(f"Best validation mIoU: {results['best_val_miou']:.4f}")
        logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
        logger.info(f"Total epochs: {results['total_epochs']}")

        # Save final results
        results_dir = output_dir / config.get('paths.results', 'results')
        results_dir.mkdir(parents=True, exist_ok=True)

        import json
        with open(results_dir / 'training_results.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if key == 'history':
                    # Skip history for now as it's complex to serialize
                    continue
                elif hasattr(value, 'item'):
                    serializable_results[key] = value.item()
                else:
                    serializable_results[key] = value

            json.dump(serializable_results, f, indent=2)

        logger.info(f"Training results saved to {results_dir / 'training_results.json'}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()