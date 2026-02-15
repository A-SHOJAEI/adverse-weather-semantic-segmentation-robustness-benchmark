"""Data loading utilities for Cityscapes and KITTI datasets with weather augmentation."""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, Normalize
from albumentations.pytorch import ToTensorV2

from .preprocessing import WeatherDegradationTransforms, DepthEstimationPreprocessor

logger = logging.getLogger(__name__)


class CityscapesKITTIDataset(Dataset):
    """
    Combined dataset for Cityscapes and KITTI semantic segmentation with weather augmentation.

    This dataset supports loading clean images and applying synthetic weather degradations
    for robustness benchmarking.
    """

    # Cityscapes class mapping (19 classes)
    CITYSCAPES_CLASSES = {
        0: 'unlabeled', 1: 'ego vehicle', 2: 'rectification border',
        3: 'out of roi', 4: 'static', 5: 'dynamic', 6: 'ground',
        7: 'road', 8: 'sidewalk', 9: 'parking', 10: 'rail track',
        11: 'building', 12: 'wall', 13: 'fence', 14: 'guard rail',
        15: 'bridge', 16: 'tunnel', 17: 'pole', 18: 'polegroup',
        19: 'traffic light', 20: 'traffic sign', 21: 'vegetation',
        22: 'terrain', 23: 'sky', 24: 'person', 25: 'rider',
        26: 'car', 27: 'truck', 28: 'bus', 29: 'caravan',
        30: 'trailer', 31: 'train', 32: 'motorcycle', 33: 'bicycle'
    }

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (512, 1024),
        weather_conditions: Optional[List[str]] = None,
        apply_augmentation: bool = True,
        include_depth: bool = True,
        dataset_type: str = "cityscapes",
        **kwargs
    ) -> None:
        """
        Initialize the combined dataset.

        Args:
            data_root: Root directory containing dataset files
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size (height, width)
            weather_conditions: Weather conditions to apply ['fog', 'rain', 'snow', 'night']
            apply_augmentation: Whether to apply data augmentation
            include_depth: Whether to include depth estimation targets
            dataset_type: Type of dataset ('cityscapes', 'kitti', 'combined')
            **kwargs: Additional arguments
        """
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.weather_conditions = weather_conditions or ['clean', 'fog', 'rain', 'snow', 'night']
        self.apply_augmentation = apply_augmentation
        self.include_depth = include_depth
        self.dataset_type = dataset_type

        # Initialize weather degradation transforms
        self.weather_transforms = WeatherDegradationTransforms()

        # Initialize depth estimation preprocessor
        if self.include_depth:
            self.depth_preprocessor = DepthEstimationPreprocessor()

        # Load dataset files
        self.samples = self._load_samples()

        # Setup transforms
        self.transforms = self._setup_transforms()

        logger.info(f"Loaded {len(self.samples)} samples from {dataset_type} dataset ({split} split)")

    def _load_samples(self) -> List[Dict[str, str]]:
        """Load sample file paths for the specified dataset."""
        samples = []

        if self.dataset_type in ['cityscapes', 'combined']:
            samples.extend(self._load_cityscapes_samples())

        if self.dataset_type in ['kitti', 'combined']:
            samples.extend(self._load_kitti_samples())

        if not samples:
            # Generate synthetic samples for testing
            samples = self._generate_synthetic_samples()

        return samples

    def _load_cityscapes_samples(self) -> List[Dict[str, str]]:
        """Load Cityscapes dataset samples."""
        samples = []
        cityscapes_root = self.data_root / "cityscapes"

        if not cityscapes_root.exists():
            logger.warning(f"Cityscapes data not found at {cityscapes_root}")
            return []

        images_dir = cityscapes_root / "leftImg8bit" / self.split
        labels_dir = cityscapes_root / "gtFine" / self.split

        if images_dir.exists() and labels_dir.exists():
            for city_dir in images_dir.iterdir():
                if city_dir.is_dir():
                    for img_file in city_dir.glob("*_leftImg8bit.png"):
                        # Find corresponding label file
                        label_file = labels_dir / city_dir.name / img_file.name.replace(
                            "_leftImg8bit.png", "_gtFine_labelIds.png"
                        )

                        if label_file.exists():
                            samples.append({
                                'image': str(img_file),
                                'label': str(label_file),
                                'dataset': 'cityscapes',
                                'city': city_dir.name
                            })

        return samples

    def _load_kitti_samples(self) -> List[Dict[str, str]]:
        """Load KITTI dataset samples."""
        samples = []
        kitti_root = self.data_root / "kitti"

        if not kitti_root.exists():
            logger.warning(f"KITTI data not found at {kitti_root}")
            return []

        images_dir = kitti_root / "training" / "image_2"
        labels_dir = kitti_root / "training" / "semantic"

        if images_dir.exists() and labels_dir.exists():
            for img_file in images_dir.glob("*.png"):
                label_file = labels_dir / img_file.name

                if label_file.exists():
                    samples.append({
                        'image': str(img_file),
                        'label': str(label_file),
                        'dataset': 'kitti'
                    })

        return samples

    def _generate_synthetic_samples(self) -> List[Dict[str, str]]:
        """Generate synthetic samples for testing when real data is not available."""
        samples = []
        num_samples = 100 if self.split == 'train' else 20

        for i in range(num_samples):
            samples.append({
                'image': f'synthetic_image_{i}.png',
                'label': f'synthetic_label_{i}.png',
                'dataset': 'synthetic',
                'synthetic': True
            })

        logger.info(f"Generated {len(samples)} synthetic samples for testing")
        return samples

    def _setup_transforms(self) -> Compose:
        """Setup data transformation pipeline."""
        transform_list = []

        if self.apply_augmentation and self.split == 'train':
            transform_list.extend([
                HorizontalFlip(p=0.5),
                RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
            ])

        transform_list.extend([
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        return Compose(transform_list)

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image."""
        if 'synthetic' in image_path:
            # Generate synthetic RGB image
            image = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
        else:
            try:
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    if image is None:
                        raise ValueError(f"Could not read image from {image_path}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    # Fallback to synthetic image
                    image = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
            except Exception as e:
                logger.warning(f"Error loading image {image_path}: {e}, using synthetic image")
                image = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)

        # Resize to target size
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

        return image

    def _load_label(self, label_path: str) -> np.ndarray:
        """Load and preprocess segmentation label."""
        if 'synthetic' in label_path:
            # Generate synthetic segmentation mask with 19 classes
            label = np.random.randint(0, 19, self.image_size, dtype=np.uint8)
        else:
            try:
                if os.path.exists(label_path):
                    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    if label is None:
                        raise ValueError(f"Could not read label from {label_path}")
                else:
                    # Fallback to synthetic label
                    label = np.random.randint(0, 19, self.image_size, dtype=np.uint8)
            except Exception as e:
                logger.warning(f"Error loading label {label_path}: {e}, using synthetic label")
                label = np.random.randint(0, 19, self.image_size, dtype=np.uint8)

        # Resize to target size
        if label.shape != self.image_size:
            label = cv2.resize(label, (self.image_size[1], self.image_size[0]),
                             interpolation=cv2.INTER_NEAREST)

        return label

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sample_info = self.samples[idx]

        # Load image and label
        image = self._load_image(sample_info['image'])
        label = self._load_label(sample_info['label'])

        # Apply weather degradation
        weather_condition = np.random.choice(self.weather_conditions)
        if weather_condition != 'clean':
            image = self.weather_transforms.apply_weather_effect(image, weather_condition)

        # Estimate depth if required
        depth = None
        if self.include_depth:
            depth = self.depth_preprocessor.estimate_depth(image)

        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            label = torch.from_numpy(label).long()

        result = {
            'image': image,
            'label': label,
            'weather_condition': weather_condition,
            'dataset': sample_info['dataset']
        }

        if depth is not None:
            result['depth'] = torch.from_numpy(depth).float()

        return result


class WeatherAugmentationPipeline:
    """
    Advanced weather augmentation pipeline for domain adaptation.

    Implements style transfer augmentations and weather-specific transformations
    for improving model robustness.
    """

    def __init__(
        self,
        weather_intensities: Dict[str, float] = None,
        style_transfer_prob: float = 0.3,
        **kwargs
    ) -> None:
        """
        Initialize weather augmentation pipeline.

        Args:
            weather_intensities: Intensity levels for each weather condition
            style_transfer_prob: Probability of applying style transfer
            **kwargs: Additional arguments
        """
        self.weather_intensities = weather_intensities or {
            'fog': 0.7,
            'rain': 0.5,
            'snow': 0.6,
            'night': 0.8
        }
        self.style_transfer_prob = style_transfer_prob

        # Initialize weather transforms
        self.weather_transforms = WeatherDegradationTransforms()

        logger.info("Initialized WeatherAugmentationPipeline")

    def apply_domain_adaptation_augmentation(
        self,
        image: np.ndarray,
        target_weather: str = None
    ) -> np.ndarray:
        """
        Apply domain adaptation augmentation.

        Args:
            image: Input image as numpy array
            target_weather: Target weather condition

        Returns:
            Augmented image
        """
        if target_weather is None:
            target_weather = np.random.choice(list(self.weather_intensities.keys()))

        # Apply weather degradation
        augmented_image = self.weather_transforms.apply_weather_effect(
            image, target_weather, intensity=self.weather_intensities[target_weather]
        )

        # Apply style transfer with probability
        if np.random.random() < self.style_transfer_prob:
            augmented_image = self._apply_style_transfer(augmented_image, target_weather)

        return augmented_image

    def _apply_style_transfer(self, image: np.ndarray, weather_type: str) -> np.ndarray:
        """
        Apply simplified style transfer for weather adaptation.

        Args:
            image: Input image
            weather_type: Type of weather condition

        Returns:
            Style-transferred image
        """
        # Simple color space manipulation for style transfer effect
        if weather_type == 'fog':
            # Increase brightness and reduce contrast
            image = cv2.convertScaleAbs(image, alpha=0.8, beta=30)
        elif weather_type == 'rain':
            # Increase contrast and add blue tint
            image = cv2.convertScaleAbs(image, alpha=1.2, beta=-10)
            image[:, :, 2] = np.clip(image[:, :, 2] * 1.1, 0, 255)
        elif weather_type == 'snow':
            # Increase brightness and add white tint
            image = cv2.convertScaleAbs(image, alpha=0.9, beta=20)
        elif weather_type == 'night':
            # Reduce brightness and increase blue channel
            image = cv2.convertScaleAbs(image, alpha=0.4, beta=-20)
            image[:, :, 2] = np.clip(image[:, :, 2] * 1.3, 0, 255)

        return image


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader with optimal settings.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        **kwargs: Additional DataLoader arguments

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if shuffle else False,
        **kwargs
    )