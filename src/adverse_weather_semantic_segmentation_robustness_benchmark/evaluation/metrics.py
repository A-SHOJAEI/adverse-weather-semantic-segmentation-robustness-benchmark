"""Evaluation metrics for robustness assessment and confidence calibration."""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import integrate

logger = logging.getLogger(__name__)


class IoUMetrics:
    """
    Intersection over Union (IoU) metrics computation.

    Provides efficient computation of per-class and mean IoU metrics
    for semantic segmentation evaluation.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        """
        Initialize IoU metrics.

        Args:
            num_classes: Number of segmentation classes
            ignore_index: Index to ignore in computation
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def compute_iou(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute IoU metrics.

        Args:
            predictions: Predicted segmentation masks [B, H, W] or [B, C, H, W]
            targets: Ground truth segmentation masks [B, H, W]

        Returns:
            Dictionary containing IoU metrics
        """
        # Handle logits input
        if predictions.dim() == 4:
            predictions = predictions.argmax(dim=1)

        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Remove ignored pixels
        valid_mask = targets != self.ignore_index
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        # Compute confusion matrix
        confusion_matrix = torch.zeros(
            self.num_classes, self.num_classes,
            dtype=torch.long, device=predictions.device
        )

        indices = targets * self.num_classes + predictions
        confusion_matrix = confusion_matrix.view(-1)
        confusion_matrix.index_add_(0, indices.long(), torch.ones_like(indices))
        confusion_matrix = confusion_matrix.view(self.num_classes, self.num_classes)

        # Compute per-class IoU
        intersection = torch.diag(confusion_matrix)
        union = confusion_matrix.sum(dim=0) + confusion_matrix.sum(dim=1) - intersection

        # Avoid division by zero
        valid_classes = union > 0
        per_class_iou = torch.zeros(self.num_classes, device=predictions.device)
        per_class_iou[valid_classes] = intersection[valid_classes] / union[valid_classes]

        # Compute mean IoU
        mean_iou = per_class_iou[valid_classes].mean()

        return {
            'mean_iou': mean_iou.item(),
            'per_class_iou': per_class_iou.cpu().numpy(),
            'valid_classes': valid_classes.cpu().numpy()
        }

    def compute_pixel_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """
        Compute pixel accuracy.

        Args:
            predictions: Predicted segmentation masks
            targets: Ground truth segmentation masks

        Returns:
            Pixel accuracy
        """
        # Handle logits input
        if predictions.dim() == 4:
            predictions = predictions.argmax(dim=1)

        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Remove ignored pixels
        valid_mask = targets != self.ignore_index
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        # Compute accuracy
        correct = (predictions == targets).sum().item()
        total = targets.numel()

        return correct / total if total > 0 else 0.0


class ConfidenceCalibration:
    """
    Confidence calibration metrics and utilities.

    Implements Expected Calibration Error (ECE) and reliability diagrams
    for assessing model confidence calibration quality.
    """

    def __init__(self, num_bins: int = 15) -> None:
        """
        Initialize confidence calibration metrics.

        Args:
            num_bins: Number of bins for calibration analysis
        """
        self.num_bins = num_bins

    def compute_ece(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        return_details: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Compute Expected Calibration Error (ECE).

        Args:
            predictions: Model predictions (logits) [B, C, H, W]
            targets: Ground truth labels [B, H, W]
            return_details: Whether to return detailed calibration information

        Returns:
            ECE value or detailed calibration metrics
        """
        # Get probabilities and confidence
        probabilities = F.softmax(predictions, dim=1)
        confidences, predicted_classes = torch.max(probabilities, dim=1)

        # Flatten spatial dimensions
        confidences = confidences.view(-1)
        predicted_classes = predicted_classes.view(-1)
        targets = targets.view(-1)

        # Filter out ignored pixels
        ignore_mask = targets == 255
        confidences = confidences[~ignore_mask]
        predicted_classes = predicted_classes[~ignore_mask]
        targets = targets[~ignore_mask]

        # Compute accuracy per sample
        accuracies = (predicted_classes == targets).float()

        # Create bins
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        bin_details = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in current bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin > 0:
                # Compute accuracy and confidence for this bin
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()

                # Compute calibration error for this bin
                bin_error = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += bin_error * prop_in_bin

                bin_details.append({
                    'bin_lower': bin_lower.item(),
                    'bin_upper': bin_upper.item(),
                    'accuracy': accuracy_in_bin.item(),
                    'confidence': avg_confidence_in_bin.item(),
                    'proportion': prop_in_bin.item(),
                    'error': bin_error.item()
                })
            else:
                bin_details.append({
                    'bin_lower': bin_lower.item(),
                    'bin_upper': bin_upper.item(),
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'proportion': 0.0,
                    'error': 0.0
                })

        if return_details:
            return {
                'ece': ece.item(),
                'bin_details': bin_details,
                'overall_accuracy': accuracies.mean().item(),
                'overall_confidence': confidences.mean().item()
            }

        return ece.item()

    def compute_reliability_diagram_data(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Compute data for reliability diagram.

        Args:
            predictions: Model predictions (logits)
            targets: Ground truth labels

        Returns:
            Dictionary containing bin data for plotting
        """
        calibration_data = self.compute_ece(predictions, targets, return_details=True)
        bin_details = calibration_data['bin_details']

        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_proportions = []

        for bin_info in bin_details:
            if bin_info['proportion'] > 0:
                bin_center = (bin_info['bin_lower'] + bin_info['bin_upper']) / 2
                bin_centers.append(bin_center)
                bin_accuracies.append(bin_info['accuracy'])
                bin_confidences.append(bin_info['confidence'])
                bin_proportions.append(bin_info['proportion'])

        return {
            'bin_centers': np.array(bin_centers),
            'bin_accuracies': np.array(bin_accuracies),
            'bin_confidences': np.array(bin_confidences),
            'bin_proportions': np.array(bin_proportions)
        }

    def temperature_scale(
        self,
        logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """
        Apply temperature scaling for calibration.

        Args:
            logits: Model logits
            temperature: Temperature parameter

        Returns:
            Temperature-scaled logits
        """
        return logits / temperature

    def optimize_temperature(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        max_iter: int = 50
    ) -> float:
        """
        Optimize temperature parameter for calibration.

        Args:
            logits: Model logits
            targets: Ground truth targets
            max_iter: Maximum optimization iterations

        Returns:
            Optimal temperature value
        """
        # Simple implementation using grid search
        temperatures = torch.linspace(0.1, 10.0, 100)
        best_temperature = 1.0
        best_nll = float('inf')

        logits_flat = logits.view(-1, logits.size(1))
        targets_flat = targets.view(-1)

        # Remove ignored pixels
        valid_mask = targets_flat != 255
        logits_flat = logits_flat[valid_mask]
        targets_flat = targets_flat[valid_mask]

        for temp in temperatures:
            scaled_logits = logits_flat / temp
            nll = F.cross_entropy(scaled_logits, targets_flat)

            if nll < best_nll:
                best_nll = nll
                best_temperature = temp.item()

        return best_temperature


class EnsembleDisagreementMetrics:
    """
    Ensemble disagreement metrics for uncertainty estimation.

    Provides metrics to assess ensemble model disagreement as a proxy
    for prediction uncertainty and model reliability.
    """

    def __init__(self) -> None:
        """Initialize ensemble disagreement metrics."""
        pass

    def compute_disagreement_map(
        self,
        predictions_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute pixel-wise disagreement map.

        Args:
            predictions_list: List of prediction tensors from ensemble models

        Returns:
            Disagreement map [B, H, W]
        """
        if len(predictions_list) < 2:
            raise ValueError("Need at least 2 predictions for disagreement computation")

        # Convert to probabilities
        prob_list = [F.softmax(pred, dim=1) for pred in predictions_list]

        # Stack predictions
        probs = torch.stack(prob_list, dim=0)  # [N_models, B, C, H, W]

        # Compute entropy of mean prediction
        mean_probs = probs.mean(dim=0)
        mean_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)

        # Compute mean of individual entropies
        individual_entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=2)
        mean_individual_entropy = individual_entropies.mean(dim=0)

        # Mutual information as disagreement measure
        disagreement = mean_entropy - mean_individual_entropy

        return disagreement

    def compute_variance_map(
        self,
        predictions_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute prediction variance map.

        Args:
            predictions_list: List of prediction tensors

        Returns:
            Variance map [B, C, H, W]
        """
        # Convert to probabilities
        prob_list = [F.softmax(pred, dim=1) for pred in predictions_list]

        # Stack and compute variance
        probs = torch.stack(prob_list, dim=0)
        variance = torch.var(probs, dim=0)

        return variance

    def compute_disagreement_auroc(
        self,
        predictions_list: List[torch.Tensor],
        targets: torch.Tensor,
        error_threshold: float = 0.5
    ) -> float:
        """
        Compute AUROC for disagreement vs. prediction errors.

        Args:
            predictions_list: List of ensemble predictions
            targets: Ground truth targets
            error_threshold: Threshold for error detection

        Returns:
            AUROC score for disagreement-error correlation
        """
        # Compute disagreement map
        disagreement = self.compute_disagreement_map(predictions_list)

        # Compute ensemble prediction
        prob_list = [F.softmax(pred, dim=1) for pred in predictions_list]
        mean_probs = torch.stack(prob_list, dim=0).mean(dim=0)
        ensemble_pred = mean_probs.argmax(dim=1)

        # Compute prediction errors
        errors = (ensemble_pred != targets).float()

        # Flatten for AUROC computation
        disagreement_flat = disagreement.view(-1).cpu().numpy()
        errors_flat = errors.view(-1).cpu().numpy()

        # Remove ignored pixels
        valid_mask = targets.view(-1) != 255
        disagreement_flat = disagreement_flat[valid_mask.cpu().numpy()]
        errors_flat = errors_flat[valid_mask.cpu().numpy()]

        if len(np.unique(errors_flat)) < 2:
            return 0.5  # No errors to predict

        # Compute AUROC
        try:
            auroc = roc_auc_score(errors_flat, disagreement_flat)
            return auroc
        except ValueError:
            return 0.5

    def compute_jensen_shannon_divergence(
        self,
        pred1: torch.Tensor,
        pred2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Jensen-Shannon divergence between two predictions.

        Args:
            pred1: First prediction tensor
            pred2: Second prediction tensor

        Returns:
            JS divergence map
        """
        # Convert to probabilities
        prob1 = F.softmax(pred1, dim=1)
        prob2 = F.softmax(pred2, dim=1)

        # Compute JS divergence
        mean_probs = (prob1 + prob2) / 2

        kl1 = F.kl_div(prob1.log(), mean_probs, reduction='none').sum(dim=1)
        kl2 = F.kl_div(prob2.log(), mean_probs, reduction='none').sum(dim=1)

        js_divergence = (kl1 + kl2) / 2

        return js_divergence


class RobustnessMetrics:
    """
    Comprehensive robustness metrics for adverse weather evaluation.

    Combines IoU metrics, confidence calibration, and ensemble disagreement
    for comprehensive robustness assessment.
    """

    def __init__(
        self,
        num_classes: int = 19,
        weather_conditions: List[str] = None
    ) -> None:
        """
        Initialize robustness metrics.

        Args:
            num_classes: Number of segmentation classes
            weather_conditions: List of weather conditions to evaluate
        """
        self.num_classes = num_classes
        self.weather_conditions = weather_conditions or ['clean', 'fog', 'rain', 'snow', 'night']

        # Initialize component metrics
        self.iou_metrics = IoUMetrics(num_classes)
        self.calibration_metrics = ConfidenceCalibration()
        self.ensemble_metrics = EnsembleDisagreementMetrics()

    def compute_miou(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """
        Compute mean IoU.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Mean IoU score
        """
        iou_results = self.iou_metrics.compute_iou(predictions, targets)
        return iou_results['mean_iou']

    def compute_weather_specific_metrics(
        self,
        predictions_dict: Dict[str, torch.Tensor],
        targets_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute metrics for each weather condition.

        Args:
            predictions_dict: Predictions grouped by weather condition
            targets_dict: Targets grouped by weather condition

        Returns:
            Weather-specific metrics
        """
        metrics = {}

        for weather in self.weather_conditions:
            if weather in predictions_dict and weather in targets_dict:
                preds = predictions_dict[weather]
                tgts = targets_dict[weather]

                if len(preds) > 0 and len(tgts) > 0:
                    miou = self.compute_miou(preds, tgts)
                    metrics[f'miou_{weather}'] = miou

        return metrics

    def compute_robustness_degradation_ratio(
        self,
        clean_miou: float,
        adverse_miou: float
    ) -> float:
        """
        Compute robustness degradation ratio.

        Args:
            clean_miou: mIoU on clean conditions
            adverse_miou: mIoU on adverse conditions

        Returns:
            Degradation ratio (0 = no degradation, 1 = complete failure)
        """
        if clean_miou == 0:
            return 1.0

        degradation = (clean_miou - adverse_miou) / clean_miou
        return max(0.0, degradation)

    def compute_comprehensive_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        ensemble_predictions: Optional[List[torch.Tensor]] = None,
        weather_condition: str = 'clean'
    ) -> Dict[str, float]:
        """
        Compute comprehensive robustness metrics.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            ensemble_predictions: List of ensemble member predictions
            weather_condition: Weather condition label

        Returns:
            Comprehensive metrics dictionary
        """
        metrics = {}

        # Basic IoU metrics
        iou_results = self.iou_metrics.compute_iou(predictions, targets)
        metrics['mean_iou'] = iou_results['mean_iou']
        metrics['pixel_accuracy'] = self.iou_metrics.compute_pixel_accuracy(predictions, targets)

        # Confidence calibration
        ece = self.calibration_metrics.compute_ece(predictions, targets)
        metrics['expected_calibration_error'] = ece

        # Ensemble disagreement metrics (if ensemble predictions available)
        if ensemble_predictions and len(ensemble_predictions) >= 2:
            disagreement_auroc = self.ensemble_metrics.compute_disagreement_auroc(
                ensemble_predictions, targets
            )
            metrics['ensemble_disagreement_auroc'] = disagreement_auroc

        # Weather-specific metric name
        metrics[f'miou_{weather_condition}'] = metrics['mean_iou']

        return metrics

    def create_robustness_summary(
        self,
        weather_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Create robustness summary from weather-specific metrics.

        Args:
            weather_metrics: Dictionary of metrics for each weather condition

        Returns:
            Summary robustness metrics
        """
        summary = {}

        # Extract mIoU scores
        clean_miou = weather_metrics.get('clean', {}).get('mean_iou', 0.0)

        # Compute degradation ratios
        for weather in ['fog', 'rain', 'snow', 'night']:
            if weather in weather_metrics:
                adverse_miou = weather_metrics[weather].get('mean_iou', 0.0)
                degradation = self.compute_robustness_degradation_ratio(clean_miou, adverse_miou)
                summary[f'robustness_degradation_{weather}'] = degradation

        # Overall degradation (mean across adverse conditions)
        degradations = [summary.get(f'robustness_degradation_{w}', 0.0)
                       for w in ['fog', 'rain', 'snow', 'night']
                       if f'robustness_degradation_{w}' in summary]

        if degradations:
            summary['robustness_degradation_ratio'] = np.mean(degradations)

        # Average ECE across conditions
        eces = [metrics.get('expected_calibration_error', 0.0)
               for metrics in weather_metrics.values()]
        if eces:
            summary['expected_calibration_error'] = np.mean(eces)

        # Average ensemble disagreement AUROC
        aurocs = [metrics.get('ensemble_disagreement_auroc', 0.5)
                 for metrics in weather_metrics.values()]
        if aurocs:
            summary['ensemble_disagreement_auroc'] = np.mean(aurocs)

        return summary