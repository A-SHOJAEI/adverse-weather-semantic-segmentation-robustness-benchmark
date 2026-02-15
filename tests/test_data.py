"""Tests for data loading and preprocessing modules."""

import pytest
import torch
import numpy as np
from pathlib import Path

from adverse_weather_semantic_segmentation_robustness_benchmark.data.loader import (
    CityscapesKITTIDataset, WeatherAugmentationPipeline, create_dataloader
)
from adverse_weather_semantic_segmentation_robustness_benchmark.data.preprocessing import (
    WeatherDegradationTransforms, DepthEstimationPreprocessor
)


class TestCityscapesKITTIDataset:
    """Test CityscapesKITTIDataset class."""

    def test_dataset_initialization(self, synthetic_dataset_dir):
        """Test dataset initialization."""
        dataset = CityscapesKITTIDataset(
            data_root=str(synthetic_dataset_dir),
            split='train',
            image_size=(256, 512),
            weather_conditions=['clean', 'fog'],
            apply_augmentation=True,
            include_depth=True,
            dataset_type='synthetic'
        )

        assert len(dataset) > 0
        assert dataset.image_size == (256, 512)
        assert 'clean' in dataset.weather_conditions
        assert 'fog' in dataset.weather_conditions

    def test_dataset_getitem(self, synthetic_dataset_dir):
        """Test dataset __getitem__ method."""
        dataset = CityscapesKITTIDataset(
            data_root=str(synthetic_dataset_dir),
            split='train',
            image_size=(128, 256),
            weather_conditions=['clean', 'fog'],
            apply_augmentation=False,  # Disable for predictable testing
            include_depth=True,
            dataset_type='synthetic'
        )

        sample = dataset[0]

        # Check required keys
        assert 'image' in sample
        assert 'label' in sample
        assert 'weather_condition' in sample
        assert 'dataset' in sample

        # Check tensor shapes and types
        assert isinstance(sample['image'], torch.Tensor)
        assert isinstance(sample['label'], torch.Tensor)
        assert sample['image'].shape == (3, 128, 256)  # C, H, W
        assert sample['label'].shape == (128, 256)  # H, W

        # Check depth if included
        if 'depth' in sample:
            assert isinstance(sample['depth'], torch.Tensor)
            assert sample['depth'].shape == (128, 256)

    def test_weather_conditions_applied(self, synthetic_dataset_dir):
        """Test that weather conditions are properly applied."""
        dataset = CityscapesKITTIDataset(
            data_root=str(synthetic_dataset_dir),
            split='train',
            image_size=(64, 128),
            weather_conditions=['fog', 'rain'],
            apply_augmentation=False,
            include_depth=False,
            dataset_type='synthetic'
        )

        # Sample multiple times to check weather variety
        weather_conditions = []
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            weather_conditions.append(sample['weather_condition'])

        # Should have some weather effects
        assert any(weather in ['fog', 'rain'] for weather in weather_conditions)

    def test_dataset_length(self, synthetic_dataset_dir):
        """Test dataset length for different splits."""
        train_dataset = CityscapesKITTIDataset(
            data_root=str(synthetic_dataset_dir),
            split='train',
            dataset_type='synthetic'
        )

        val_dataset = CityscapesKITTIDataset(
            data_root=str(synthetic_dataset_dir),
            split='val',
            dataset_type='synthetic'
        )

        assert len(train_dataset) > 0
        assert len(val_dataset) > 0
        # Training set should typically be larger
        assert len(train_dataset) >= len(val_dataset)


class TestWeatherAugmentationPipeline:
    """Test WeatherAugmentationPipeline class."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = WeatherAugmentationPipeline(
            weather_intensities={'fog': 0.5, 'rain': 0.3},
            style_transfer_prob=0.2
        )

        assert pipeline.weather_intensities['fog'] == 0.5
        assert pipeline.weather_intensities['rain'] == 0.3
        assert pipeline.style_transfer_prob == 0.2

    def test_domain_adaptation_augmentation(self):
        """Test domain adaptation augmentation."""
        pipeline = WeatherAugmentationPipeline()

        # Create test image
        test_image = np.random.randint(0, 255, (128, 256, 3), dtype=np.uint8)

        # Apply augmentation
        augmented = pipeline.apply_domain_adaptation_augmentation(
            test_image, target_weather='fog'
        )

        # Check output properties
        assert augmented.shape == test_image.shape
        assert augmented.dtype == np.uint8
        assert np.all(augmented >= 0) and np.all(augmented <= 255)


class TestWeatherDegradationTransforms:
    """Test WeatherDegradationTransforms class."""

    def test_transforms_initialization(self):
        """Test transforms initialization."""
        transforms = WeatherDegradationTransforms(seed=42)
        assert transforms.fog_parameters is not None
        assert transforms.rain_parameters is not None
        assert transforms.snow_parameters is not None
        assert transforms.night_parameters is not None

    def test_clean_weather_passthrough(self):
        """Test that clean weather returns unchanged image."""
        transforms = WeatherDegradationTransforms()
        test_image = np.random.randint(0, 255, (64, 128, 3), dtype=np.uint8)

        result = transforms.apply_weather_effect(test_image, 'clean')

        np.testing.assert_array_equal(result, test_image)

    def test_fog_effect(self):
        """Test fog weather effect."""
        transforms = WeatherDegradationTransforms(seed=42)
        test_image = np.random.randint(0, 255, (64, 128, 3), dtype=np.uint8)

        fogged = transforms.apply_weather_effect(test_image, 'fog', intensity=0.5)

        # Check output properties
        assert fogged.shape == test_image.shape
        assert fogged.dtype == np.uint8
        assert np.all(fogged >= 0) and np.all(fogged <= 255)

        # Fog typically increases brightness and reduces contrast
        # (difficult to test exactly due to randomness, but shape should be preserved)

    def test_rain_effect(self):
        """Test rain weather effect."""
        transforms = WeatherDegradationTransforms(seed=42)
        test_image = np.random.randint(0, 255, (64, 128, 3), dtype=np.uint8)

        rainy = transforms.apply_weather_effect(test_image, 'rain', intensity=0.5)

        assert rainy.shape == test_image.shape
        assert rainy.dtype == np.uint8
        assert np.all(rainy >= 0) and np.all(rainy <= 255)

    def test_snow_effect(self):
        """Test snow weather effect."""
        transforms = WeatherDegradationTransforms(seed=42)
        test_image = np.random.randint(0, 255, (64, 128, 3), dtype=np.uint8)

        snowy = transforms.apply_weather_effect(test_image, 'snow', intensity=0.5)

        assert snowy.shape == test_image.shape
        assert snowy.dtype == np.uint8
        assert np.all(snowy >= 0) and np.all(snowy <= 255)

    def test_night_effect(self):
        """Test night weather effect."""
        transforms = WeatherDegradationTransforms(seed=42)
        test_image = np.random.randint(0, 255, (64, 128, 3), dtype=np.uint8)

        night = transforms.apply_weather_effect(test_image, 'night', intensity=0.5)

        assert night.shape == test_image.shape
        assert night.dtype == np.uint8
        assert np.all(night >= 0) and np.all(night <= 255)

    def test_invalid_weather_type(self):
        """Test error handling for invalid weather type."""
        transforms = WeatherDegradationTransforms()
        test_image = np.random.randint(0, 255, (64, 128, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Unknown weather type"):
            transforms.apply_weather_effect(test_image, 'invalid_weather')

    def test_fog_density_map_generation(self):
        """Test fog density map generation."""
        transforms = WeatherDegradationTransforms(seed=42)
        test_image = np.random.rand(64, 128, 3)

        fog_density = transforms.get_fog_density_map(test_image)

        assert fog_density.shape == (64, 128)
        assert np.all(fog_density >= 0) and np.all(fog_density <= 1)


class TestDepthEstimationPreprocessor:
    """Test DepthEstimationPreprocessor class."""

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DepthEstimationPreprocessor()
        assert preprocessor is not None

    def test_depth_estimation(self):
        """Test depth estimation from RGB image."""
        preprocessor = DepthEstimationPreprocessor()
        test_image = np.random.randint(0, 255, (64, 128, 3), dtype=np.uint8)

        depth = preprocessor.estimate_depth(test_image)

        assert depth.shape == (64, 128)
        assert np.all(depth >= 0) and np.all(depth <= 1)

    def test_depth_to_disparity_conversion(self):
        """Test depth to disparity conversion."""
        preprocessor = DepthEstimationPreprocessor()
        test_depth = np.random.uniform(0.1, 10.0, (64, 128))

        disparity = preprocessor.depth_to_disparity(test_depth, baseline=0.54)

        assert disparity.shape == test_depth.shape
        assert np.all(disparity > 0)  # Disparity should be positive

    def test_depth_preprocessing_for_training(self):
        """Test depth preprocessing for training."""
        preprocessor = DepthEstimationPreprocessor()
        test_depth = np.random.uniform(0, 10, (64, 128))
        target_size = (64, 128)

        depth_tensor = preprocessor.preprocess_depth_for_training(test_depth, target_size)

        assert isinstance(depth_tensor, torch.Tensor)
        assert depth_tensor.shape == target_size
        assert torch.all(depth_tensor >= 0) and torch.all(depth_tensor <= 1)


class TestDataLoader:
    """Test data loader creation and functionality."""

    def test_dataloader_creation(self, synthetic_dataset_dir):
        """Test data loader creation."""
        dataset = CityscapesKITTIDataset(
            data_root=str(synthetic_dataset_dir),
            split='train',
            image_size=(64, 128),
            dataset_type='synthetic'
        )

        dataloader = create_dataloader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        assert len(dataloader) > 0

        # Test iteration
        for batch in dataloader:
            assert 'image' in batch
            assert 'label' in batch
            assert batch['image'].shape[0] <= 2  # Batch size
            break  # Just test first batch

    def test_dataloader_batch_consistency(self, synthetic_dataset_dir):
        """Test that dataloader produces consistent batches."""
        dataset = CityscapesKITTIDataset(
            data_root=str(synthetic_dataset_dir),
            split='train',
            image_size=(64, 128),
            dataset_type='synthetic'
        )

        dataloader = create_dataloader(
            dataset,
            batch_size=2,
            shuffle=False,  # No shuffle for consistency
            num_workers=0
        )

        batch = next(iter(dataloader))

        # Check batch structure
        assert isinstance(batch, dict)
        assert 'image' in batch and 'label' in batch

        # Check tensor properties
        images = batch['image']
        labels = batch['label']

        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert images.dim() == 4  # [B, C, H, W]
        assert labels.dim() == 3  # [B, H, W]
        assert images.shape[0] == labels.shape[0]  # Same batch size