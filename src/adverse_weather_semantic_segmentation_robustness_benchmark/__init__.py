"""
Adverse Weather Semantic Segmentation Robustness Benchmark.

A comprehensive benchmarking framework for evaluating semantic segmentation
model robustness under synthetically degraded weather conditions.
"""

__version__ = "1.0.0"
__author__ = "Alireza Shojaei"

# Import basic utilities that don't require torch
from .utils.config import Config

# Conditional imports for torch-dependent modules
try:
    from .models.model import (
        SegFormerModel,
        DeepLabV3PlusModel,
        EnsembleModel,
        FogDensityAwareLoss,
    )
    from .training.trainer import AdverseWeatherTrainer
    from .evaluation.metrics import RobustnessMetrics
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # Create dummy classes for graceful fallback
    class SegFormerModel: pass
    class DeepLabV3PlusModel: pass
    class EnsembleModel: pass
    class FogDensityAwareLoss: pass
    class AdverseWeatherTrainer: pass
    class RobustnessMetrics: pass

__all__ = [
    "SegFormerModel",
    "DeepLabV3PlusModel",
    "EnsembleModel",
    "FogDensityAwareLoss",
    "AdverseWeatherTrainer",
    "RobustnessMetrics",
    "Config",
]

# Add availability flag
__all__.append("_TORCH_AVAILABLE")