#!/usr/bin/env python3
"""
Evaluation script for adverse weather semantic segmentation models.

This script evaluates trained models on test data and computes comprehensive
robustness metrics including mIoU, calibration error, and ensemble disagreement.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import json
from typing import Dict, List, Any

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adverse_weather_semantic_segmentation_robustness_benchmark.utils.config import (
    load_config, create_default_config, setup_logging, get_device_config
)
from adverse_weather_semantic_segmentation_robustness_benchmark.data.loader import (
    CityscapesKITTIDataset, create_dataloader
)
from adverse_weather_semantic_segmentation_robustness_benchmark.models.model import (
    SegFormerModel, DeepLabV3PlusModel, EnsembleModel
)
from adverse_weather_semantic_segmentation_robustness_benchmark.evaluation.metrics import (
    RobustnessMetrics, ConfidenceCalibration, EnsembleDisagreementMetrics
)

logger = logging.getLogger(__name__)


def load_model(config, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load model from checkpoint.

    Args:
        config: Configuration object
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model
    """
    # Create model
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

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"Loaded {model_type} model from {checkpoint_path}")
    return model


def create_test_dataset_and_loader(config):
    """
    Create test dataset and data loader.

    Args:
        config: Configuration object

    Returns:
        Test data loader
    """
    data_root = config.get('data.data_root', 'data')
    dataset_type = config.get('data.dataset_type', 'combined')
    image_size = tuple(config.get('data.image_size', [512, 1024]))
    weather_conditions = config.get('data.weather_conditions', ['clean', 'fog', 'rain', 'snow', 'night'])
    include_depth = config.get('data.include_depth', True)

    batch_size = config.get('training.batch_size', 8)
    num_workers = config.get('training.num_workers', 4)

    # Create test dataset
    test_dataset = CityscapesKITTIDataset(
        data_root=data_root,
        split='test',  # Use test split for evaluation
        image_size=image_size,
        weather_conditions=weather_conditions,
        apply_augmentation=False,  # No augmentation for evaluation
        include_depth=include_depth,
        dataset_type=dataset_type
    )

    # Create data loader
    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(f"Created test dataset with {len(test_dataset)} samples")
    logger.info(f"Created test dataloader with {len(test_loader)} batches")

    return test_loader


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    metrics: RobustnessMetrics,
    device: torch.device,
    config
) -> Dict[str, Any]:
    """
    Evaluate model on test data.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        metrics: Metrics calculator
        device: Device for evaluation
        config: Configuration object

    Returns:
        Evaluation results
    """
    model.eval()

    # Initialize result containers
    all_predictions = []
    all_targets = []
    all_logits = []
    weather_predictions = {weather: [] for weather in config.get('data.weather_conditions', [])}
    weather_targets = {weather: [] for weather in config.get('data.weather_conditions', [])}
    weather_logits = {weather: [] for weather in config.get('data.weather_conditions', [])}

    # Ensemble predictions (if ensemble model)
    ensemble_predictions = []
    individual_predictions = {'segformer': [], 'deeplabv3plus': []}

    logger.info("Starting model evaluation...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            weather_conditions = batch.get('weather_condition', ['clean'] * images.size(0))

            # Forward pass
            outputs = model(images)
            logits = outputs['segmentation']
            predictions = logits.argmax(dim=1)

            # Collect overall results
            all_predictions.append(predictions.cpu())
            all_targets.append(labels.cpu())
            all_logits.append(logits.cpu())

            # Group by weather condition
            for i, weather in enumerate(weather_conditions):
                if weather in weather_predictions:
                    weather_predictions[weather].append(predictions[i:i+1].cpu())
                    weather_targets[weather].append(labels[i:i+1].cpu())
                    weather_logits[weather].append(logits[i:i+1].cpu())

            # Collect ensemble member predictions if available
            if hasattr(model, 'segformer') and 'segformer_seg' in outputs:
                individual_predictions['segformer'].append(outputs['segformer_seg'].cpu())
                individual_predictions['deeplabv3plus'].append(outputs['deeplabv3plus_seg'].cpu())
                ensemble_predictions.append([
                    outputs['segformer_seg'].cpu(),
                    outputs['deeplabv3plus_seg'].cpu()
                ])

    # Concatenate results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_logits = torch.cat(all_logits, dim=0)

    # Concatenate weather-specific results
    for weather in weather_predictions:
        if weather_predictions[weather]:
            weather_predictions[weather] = torch.cat(weather_predictions[weather], dim=0)
            weather_targets[weather] = torch.cat(weather_targets[weather], dim=0)
            weather_logits[weather] = torch.cat(weather_logits[weather], dim=0)

    # Compute overall metrics
    results = {}

    # Overall mIoU
    overall_miou = metrics.compute_miou(all_predictions, all_targets)
    results['overall_miou'] = overall_miou

    # Weather-specific mIoU
    weather_mious = {}
    for weather, preds in weather_predictions.items():
        if len(preds) > 0:
            weather_miou = metrics.compute_miou(preds, weather_targets[weather])
            weather_mious[weather] = weather_miou
            results[f'miou_{weather}'] = weather_miou

    # Confidence calibration metrics
    calibration_metrics = ConfidenceCalibration()
    overall_ece = calibration_metrics.compute_ece(all_logits, all_targets)
    results['expected_calibration_error'] = overall_ece

    # Weather-specific ECE
    for weather, logits in weather_logits.items():
        if len(logits) > 0:
            weather_ece = calibration_metrics.compute_ece(logits, weather_targets[weather])
            results[f'ece_{weather}'] = weather_ece

    # Ensemble disagreement metrics (if ensemble model)
    if ensemble_predictions:
        ensemble_metrics = EnsembleDisagreementMetrics()

        # Convert list of pairs to list of lists
        segformer_preds = individual_predictions['segformer']
        deeplabv3plus_preds = individual_predictions['deeplabv3plus']

        if segformer_preds and deeplabv3plus_preds:
            segformer_all = torch.cat(segformer_preds, dim=0)
            deeplabv3plus_all = torch.cat(deeplabv3plus_preds, dim=0)

            ensemble_auroc = ensemble_metrics.compute_disagreement_auroc(
                [segformer_all, deeplabv3plus_all], all_targets
            )
            results['ensemble_disagreement_auroc'] = ensemble_auroc

    # Compute robustness degradation ratios
    if 'clean' in weather_mious:
        clean_miou = weather_mious['clean']
        for weather in ['fog', 'rain', 'snow', 'night']:
            if weather in weather_mious:
                adverse_miou = weather_mious[weather]
                degradation = metrics.compute_robustness_degradation_ratio(clean_miou, adverse_miou)
                results[f'robustness_degradation_{weather}'] = degradation

        # Overall robustness degradation
        degradations = [results.get(f'robustness_degradation_{w}', 0.0)
                       for w in ['fog', 'rain', 'snow', 'night']
                       if f'robustness_degradation_{w}' in results]
        if degradations:
            results['robustness_degradation_ratio'] = np.mean(degradations)

    logger.info("Model evaluation completed")
    return results


def generate_evaluation_report(
    results: Dict[str, Any],
    output_dir: Path,
    target_metrics: Dict[str, float] = None
) -> None:
    """
    Generate comprehensive evaluation report.

    Args:
        results: Evaluation results
        output_dir: Output directory for report
        target_metrics: Target metrics for comparison
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Create summary table
    summary_data = []

    # Target metrics from specification
    if target_metrics is None:
        target_metrics = {
            'miou_clean': 0.78,
            'miou_fog': 0.65,
            'miou_rain': 0.62,
            'robustness_degradation_ratio': 0.18,
            'expected_calibration_error': 0.05,
            'ensemble_disagreement_auroc': 0.85
        }

    for metric, target_value in target_metrics.items():
        actual_value = results.get(metric, 0.0)
        status = "✓" if actual_value >= target_value else "✗"
        summary_data.append({
            'Metric': metric,
            'Target': f"{target_value:.3f}",
            'Actual': f"{actual_value:.3f}",
            'Status': status
        })

    # Create results summary
    report_lines = [
        "# Adverse Weather Semantic Segmentation Evaluation Report",
        "",
        "## Summary Metrics",
        ""
    ]

    # Add summary table
    report_lines.extend([
        "| Metric | Target | Actual | Status |",
        "|--------|--------|--------|--------|"
    ])

    for row in summary_data:
        report_lines.append(f"| {row['Metric']} | {row['Target']} | {row['Actual']} | {row['Status']} |")

    # Weather-specific performance
    report_lines.extend([
        "",
        "## Weather-Specific Performance",
        ""
    ])

    weather_conditions = ['clean', 'fog', 'rain', 'snow', 'night']
    for weather in weather_conditions:
        miou_key = f'miou_{weather}'
        if miou_key in results:
            miou = results[miou_key]
            report_lines.append(f"- **{weather.title()}**: mIoU = {miou:.3f}")

    # Robustness analysis
    report_lines.extend([
        "",
        "## Robustness Analysis",
        ""
    ])

    if 'robustness_degradation_ratio' in results:
        degradation = results['robustness_degradation_ratio']
        report_lines.append(f"- **Overall Degradation Ratio**: {degradation:.3f}")

    for weather in ['fog', 'rain', 'snow', 'night']:
        deg_key = f'robustness_degradation_{weather}'
        if deg_key in results:
            degradation = results[deg_key]
            report_lines.append(f"- **{weather.title()} Degradation**: {degradation:.3f}")

    # Confidence calibration
    if 'expected_calibration_error' in results:
        ece = results['expected_calibration_error']
        report_lines.extend([
            "",
            "## Confidence Calibration",
            "",
            f"- **Expected Calibration Error**: {ece:.3f}"
        ])

    # Ensemble performance
    if 'ensemble_disagreement_auroc' in results:
        auroc = results['ensemble_disagreement_auroc']
        report_lines.extend([
            "",
            "## Ensemble Performance",
            "",
            f"- **Disagreement AUROC**: {auroc:.3f}"
        ])

    # Write report
    with open(output_dir / 'evaluation_report.md', 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Evaluation report saved to {output_dir}")


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate adverse weather semantic segmentation models")
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cpu, cuda, cuda:0, etc.)'
    )

    args = parser.parse_args()

    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

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

    # Override device if specified
    if args.device != 'auto':
        config.set('device', args.device)

    # Setup logging
    setup_logging(config)

    # Setup device
    device_str = get_device_config(config.get('device', 'auto'))
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    try:
        model = load_model(config, args.checkpoint, device)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

    # Create test dataset and loader
    try:
        test_loader = create_test_dataset_and_loader(config)
    except Exception as e:
        logger.error(f"Error creating test dataset: {e}")
        logger.info("This error is expected if real dataset files are not available.")
        logger.info("The system will use synthetic data for evaluation.")
        test_loader = create_test_dataset_and_loader(config)

    # Initialize metrics
    metrics = RobustnessMetrics(
        num_classes=config.get('model.num_classes', 19),
        weather_conditions=config.get('data.weather_conditions', ['clean', 'fog', 'rain', 'snow', 'night'])
    )

    # Evaluate model
    try:
        results = evaluate_model(model, test_loader, metrics, device, config)

        # Print key results
        logger.info("=== Evaluation Results ===")
        logger.info(f"Overall mIoU: {results.get('overall_miou', 0.0):.4f}")

        for weather in ['clean', 'fog', 'rain', 'snow', 'night']:
            miou_key = f'miou_{weather}'
            if miou_key in results:
                logger.info(f"mIoU ({weather}): {results[miou_key]:.4f}")

        if 'robustness_degradation_ratio' in results:
            logger.info(f"Robustness Degradation Ratio: {results['robustness_degradation_ratio']:.4f}")

        if 'expected_calibration_error' in results:
            logger.info(f"Expected Calibration Error: {results['expected_calibration_error']:.4f}")

        if 'ensemble_disagreement_auroc' in results:
            logger.info(f"Ensemble Disagreement AUROC: {results['ensemble_disagreement_auroc']:.4f}")

        # Generate evaluation report
        generate_evaluation_report(results, output_dir)

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()