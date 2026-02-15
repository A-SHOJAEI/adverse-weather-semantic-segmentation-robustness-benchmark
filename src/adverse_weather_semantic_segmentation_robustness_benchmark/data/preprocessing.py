"""Data preprocessing utilities for weather degradation and depth estimation."""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


class WeatherDegradationTransforms:
    """
    Weather degradation transforms for synthetic adverse conditions.

    Implements physically-based weather simulation including fog, rain, snow,
    and nighttime conditions with realistic parameter ranges.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize weather degradation transforms.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        self.fog_parameters = {
            'beta_range': (0.005, 0.05),  # Atmospheric scattering coefficient
            'A_range': (0.7, 1.0),        # Atmospheric light
            'depth_scale': 100.0          # Depth scaling factor
        }

        self.rain_parameters = {
            'intensity_range': (0.1, 0.8),
            'drop_size_range': (1, 3),
            'angle_range': (-15, 15),
            'num_drops_range': (100, 500)
        }

        self.snow_parameters = {
            'intensity_range': (0.1, 0.7),
            'flake_size_range': (2, 8),
            'num_flakes_range': (50, 200),
            'blur_kernel': (3, 7)
        }

        self.night_parameters = {
            'brightness_reduction': (0.2, 0.6),
            'color_shift': {'r': 0.8, 'g': 0.85, 'b': 1.2},
            'noise_std': 5.0
        }

        logger.info("Initialized WeatherDegradationTransforms")

    def apply_weather_effect(
        self,
        image: np.ndarray,
        weather_type: str,
        intensity: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply weather degradation effect to image.

        Args:
            image: Input image as numpy array (H, W, C)
            weather_type: Type of weather ('fog', 'rain', 'snow', 'night')
            intensity: Weather intensity (0.0 to 1.0), if None uses random

        Returns:
            Weather-degraded image
        """
        if weather_type == 'clean':
            return image

        image = image.astype(np.float32) / 255.0

        if weather_type == 'fog':
            return self._apply_fog(image, intensity)
        elif weather_type == 'rain':
            return self._apply_rain(image, intensity)
        elif weather_type == 'snow':
            return self._apply_snow(image, intensity)
        elif weather_type == 'night':
            return self._apply_night(image, intensity)
        else:
            raise ValueError(f"Unknown weather type: {weather_type}")

    def _apply_fog(self, image: np.ndarray, intensity: Optional[float] = None) -> np.ndarray:
        """
        Apply fog effect using atmospheric scattering model.

        Implements: I_fog = I * exp(-beta * d) + A * (1 - exp(-beta * d))
        Where beta is scattering coefficient, d is depth, A is atmospheric light.
        """
        h, w = image.shape[:2]

        # Generate synthetic depth map (increasing with distance)
        depth = self._generate_synthetic_depth(h, w)

        # Sample fog parameters
        if intensity is None:
            intensity = np.random.uniform(0.3, 0.9)

        beta_min, beta_max = self.fog_parameters['beta_range']
        A_min, A_max = self.fog_parameters['A_range']

        beta = beta_min + intensity * (beta_max - beta_min)
        A = A_min + intensity * (A_max - A_min)

        # Apply atmospheric scattering model
        transmission = np.exp(-beta * depth)
        atmospheric_light = A * np.ones_like(image)

        fogged_image = image * transmission[..., np.newaxis] + \
                      atmospheric_light * (1 - transmission[..., np.newaxis])

        return (np.clip(fogged_image, 0, 1) * 255).astype(np.uint8)

    def _apply_rain(self, image: np.ndarray, intensity: Optional[float] = None) -> np.ndarray:
        """Apply rain effect with realistic raindrops and atmosphere."""
        if intensity is None:
            intensity = np.random.uniform(0.2, 0.8)

        h, w = image.shape[:2]
        rain_image = image.copy()

        # Add atmospheric haze
        haze_intensity = intensity * 0.3
        rain_image = rain_image * (1 - haze_intensity) + haze_intensity * 0.7

        # Generate raindrops
        num_drops = int(self.rain_parameters['num_drops_range'][0] +
                       intensity * (self.rain_parameters['num_drops_range'][1] -
                                   self.rain_parameters['num_drops_range'][0]))

        for _ in range(num_drops):
            # Random raindrop parameters
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            length = np.random.randint(5, 20)
            thickness = np.random.choice(self.rain_parameters['drop_size_range'])
            angle = np.random.uniform(*self.rain_parameters['angle_range'])

            # Calculate end point
            end_x = int(x + length * np.sin(np.radians(angle)))
            end_y = int(y + length * np.cos(np.radians(angle)))

            # Ensure coordinates are within bounds
            end_x = np.clip(end_x, 0, w - 1)
            end_y = np.clip(end_y, 0, h - 1)

            # Draw raindrop
            rain_color = [0.8, 0.9, 1.0]  # Slightly blue-white
            cv2.line(
                rain_image, (x, y), (end_x, end_y),
                rain_color, thickness
            )

        # Add slight blur for realism
        rain_image = cv2.GaussianBlur(rain_image, (3, 3), 0.5)

        return (np.clip(rain_image, 0, 1) * 255).astype(np.uint8)

    def _apply_snow(self, image: np.ndarray, intensity: Optional[float] = None) -> np.ndarray:
        """Apply snow effect with falling snowflakes."""
        if intensity is None:
            intensity = np.random.uniform(0.2, 0.7)

        h, w = image.shape[:2]
        snow_image = image.copy()

        # Add atmospheric brightness (snow reflects light)
        brightness_boost = intensity * 0.2
        snow_image = np.clip(snow_image + brightness_boost, 0, 1)

        # Generate snowflakes
        num_flakes = int(self.snow_parameters['num_flakes_range'][0] +
                        intensity * (self.snow_parameters['num_flakes_range'][1] -
                                   self.snow_parameters['num_flakes_range'][0]))

        for _ in range(num_flakes):
            # Random snowflake parameters
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            size = np.random.choice(self.snow_parameters['flake_size_range'])

            # Draw snowflake as white circle
            cv2.circle(snow_image, (x, y), size, (1.0, 1.0, 1.0), -1)

        # Add slight blur for depth effect
        blur_kernel = np.random.choice(self.snow_parameters['blur_kernel'])
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        snow_image = cv2.GaussianBlur(snow_image, (blur_kernel, blur_kernel), 1.0)

        return (np.clip(snow_image, 0, 1) * 255).astype(np.uint8)

    def _apply_night(self, image: np.ndarray, intensity: Optional[float] = None) -> np.ndarray:
        """Apply nighttime effect with reduced brightness and color shift."""
        if intensity is None:
            intensity = np.random.uniform(0.4, 0.8)

        night_image = image.copy()

        # Reduce overall brightness
        brightness_factor = 1 - intensity * np.random.uniform(*self.night_parameters['brightness_reduction'])
        night_image = night_image * brightness_factor

        # Apply color shift (cooler tones)
        color_shift = self.night_parameters['color_shift']
        night_image[:, :, 0] *= color_shift['r']  # Red
        night_image[:, :, 1] *= color_shift['g']  # Green
        night_image[:, :, 2] *= color_shift['b']  # Blue

        # Add slight noise for realism
        noise = np.random.normal(0, self.night_parameters['noise_std'] / 255.0, night_image.shape)
        night_image = night_image + noise * intensity * 0.5

        return (np.clip(night_image, 0, 1) * 255).astype(np.uint8)

    def _generate_synthetic_depth(self, height: int, width: int) -> np.ndarray:
        """
        Generate synthetic depth map for fog simulation.

        Creates a realistic depth pattern with closer objects at bottom
        and distant objects at top (typical road scene layout).
        """
        # Create base depth gradient (top = far, bottom = near)
        y_coords = np.arange(height)[:, np.newaxis]
        depth_base = (y_coords / height) * self.fog_parameters['depth_scale']

        # Add some variation for realism
        noise = np.random.normal(0, 10, (height, width))
        depth = depth_base + noise

        # Smooth the depth map
        depth = gaussian_filter(depth, sigma=2)

        # Ensure positive depth values
        depth = np.maximum(depth, 1.0)

        return depth

    def get_fog_density_map(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Estimate fog density for the fog-density-aware loss function.

        Args:
            image: Input image
            depth: Depth map (if available)

        Returns:
            Fog density map (0 = clear, 1 = heavy fog)
        """
        h, w = image.shape[:2]

        if depth is None:
            depth = self._generate_synthetic_depth(h, w)

        # Estimate fog density based on contrast reduction
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32) / 255.0

        # Calculate local contrast
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray, -1, kernel)
        local_variance = cv2.filter2D((gray - local_mean) ** 2, -1, kernel)
        local_contrast = np.sqrt(local_variance)

        # High fog density corresponds to low contrast
        max_contrast = np.percentile(local_contrast, 95)
        fog_density = 1.0 - (local_contrast / (max_contrast + 1e-8))

        # Combine with depth information
        normalized_depth = depth / np.max(depth)
        fog_density = fog_density * (0.3 + 0.7 * normalized_depth)

        return np.clip(fog_density, 0, 1)


class DepthEstimationPreprocessor:
    """
    Depth estimation preprocessor for scene understanding.

    Provides simplified depth estimation for fog-density-aware loss computation
    and weather effect simulation.
    """

    def __init__(self) -> None:
        """Initialize depth estimation preprocessor."""
        self.depth_model = None  # Placeholder for actual depth model
        logger.info("Initialized DepthEstimationPreprocessor")

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from RGB image.

        Args:
            image: Input RGB image

        Returns:
            Estimated depth map (normalized 0-1)
        """
        # Simplified depth estimation using geometric assumptions
        # In a real implementation, you would use a pretrained depth estimation model
        h, w = image.shape[:2]

        # Generate depth based on image structure
        depth = self._geometric_depth_estimation(image)

        return depth

    def _geometric_depth_estimation(self, image: np.ndarray) -> np.ndarray:
        """
        Simple geometric depth estimation based on typical road scene assumptions.

        Args:
            image: Input RGB image

        Returns:
            Estimated depth map
        """
        h, w = image.shape[:2]

        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Sky detection (typically upper part of image)
        sky_mask = np.zeros((h, w), dtype=np.float32)
        sky_mask[:h//3, :] = 1.0

        # Road detection (typically lower part of image)
        road_mask = np.zeros((h, w), dtype=np.float32)
        road_mask[h//2:, :] = 1.0

        # Base depth from vertical position (perspective)
        y_coords = np.arange(h)[:, np.newaxis] / h
        base_depth = np.tile(y_coords * 0.8 + 0.2, (1, w))

        # Adjust depth based on detected regions
        depth = base_depth.copy()
        depth[sky_mask > 0] = 1.0  # Sky is infinitely far
        depth[road_mask > 0] *= 0.5  # Road is closer

        # Add texture-based depth cues
        # High-frequency content often indicates closer objects
        texture = cv2.Laplacian(gray, cv2.CV_64F)
        texture_strength = np.abs(texture) / (np.max(np.abs(texture)) + 1e-8)

        # Closer objects have more texture detail
        texture_depth_adjustment = -0.3 * texture_strength
        depth = np.clip(depth + texture_depth_adjustment, 0, 1)

        # Smooth the depth map
        depth = gaussian_filter(depth, sigma=2)

        return depth

    def depth_to_disparity(self, depth: np.ndarray, baseline: float = 0.54) -> np.ndarray:
        """
        Convert depth to disparity for stereo vision applications.

        Args:
            depth: Depth map
            baseline: Stereo baseline in meters

        Returns:
            Disparity map
        """
        # Prevent division by zero
        depth_safe = np.maximum(depth, 1e-6)
        disparity = baseline / depth_safe

        return disparity

    def preprocess_depth_for_training(
        self,
        depth: np.ndarray,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Preprocess depth map for training.

        Args:
            depth: Raw depth map
            target_size: Target size (height, width)

        Returns:
            Preprocessed depth tensor
        """
        # Resize if necessary
        if depth.shape != target_size:
            depth = cv2.resize(depth, (target_size[1], target_size[0]))

        # Normalize to [0, 1] range
        depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-8)

        # Convert to tensor
        depth_tensor = torch.from_numpy(depth_normalized).float()

        return depth_tensor