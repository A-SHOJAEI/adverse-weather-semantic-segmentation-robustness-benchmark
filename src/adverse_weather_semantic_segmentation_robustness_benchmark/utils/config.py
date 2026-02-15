"""Configuration management utilities."""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration management class.

    Provides unified interface for loading and managing configuration
    from YAML files with environment variable override support.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize configuration.

        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict or {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support.

        Args:
            key: Configuration key (supports dot notation like 'model.num_classes')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value with dot notation support.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config

        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the final value
        config[keys[-1]] = value

    def update(self, other_config: Union['Config', Dict[str, Any]]) -> None:
        """
        Update configuration with another config or dictionary.

        Args:
            other_config: Configuration to merge
        """
        if isinstance(other_config, Config):
            other_dict = other_config._config
        else:
            other_dict = other_config

        self._config = self._deep_merge(self._config, other_dict)

    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.

        Args:
            dict1: First dictionary
            dict2: Second dictionary

        Returns:
            Merged dictionary
        """
        result = dict1.copy()

        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return self._config.copy()

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value using bracket notation."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if configuration contains key."""
        return self.get(key) is not None

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self._config})"


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # Apply environment variable overrides
        config_dict = _apply_env_overrides(config_dict)

        config = Config(config_dict)
        logger.info(f"Loaded configuration from {config_path}")

        return config

    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading configuration from {config_path}: {e}")


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object to save
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config.to_dict(), f, default_flow_style=False, indent=2)

        logger.info(f"Saved configuration to {config_path}")

    except Exception as e:
        raise RuntimeError(f"Error saving configuration to {config_path}: {e}")


def _apply_env_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to configuration.

    Environment variables should use the format:
    CONFIG_SECTION__SUBSECTION__KEY=value

    Args:
        config_dict: Original configuration dictionary

    Returns:
        Configuration with environment overrides applied
    """
    env_prefix = "CONFIG_"

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(env_prefix):
            continue

        # Convert environment key to config key
        config_key = env_key[len(env_prefix):].lower().replace('__', '.')

        # Parse environment value
        parsed_value = _parse_env_value(env_value)

        # Set the value in config
        _set_nested_value(config_dict, config_key, parsed_value)

        logger.debug(f"Applied environment override: {config_key} = {parsed_value}")

    return config_dict


def _parse_env_value(value: str) -> Union[str, int, float, bool]:
    """
    Parse environment variable value to appropriate type.

    Args:
        value: String value from environment variable

    Returns:
        Parsed value with appropriate type
    """
    # Try boolean
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'

    # Try integer
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string
    return value


def _set_nested_value(config_dict: Dict[str, Any], key: str, value: Any) -> None:
    """
    Set nested value in configuration dictionary using dot notation.

    Args:
        config_dict: Configuration dictionary to modify
        key: Dot-separated key path
        value: Value to set
    """
    keys = key.split('.')
    current = config_dict

    # Navigate to parent dictionary
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    # Set final value
    current[keys[-1]] = value


def create_default_config() -> Config:
    """
    Create default configuration for the project.

    Returns:
        Default configuration object
    """
    default_config = {
        'model': {
            'type': 'ensemble',
            'num_classes': 19,
            'include_depth': True,
            'ensemble_strategy': 'weighted_average',
            'temperature_scaling': True
        },
        'data': {
            'dataset_type': 'combined',
            'data_root': 'data',
            'image_size': [512, 1024],
            'weather_conditions': ['clean', 'fog', 'rain', 'snow', 'night'],
            'apply_augmentation': True,
            'include_depth': True
        },
        'training': {
            'batch_size': 2,
            'epochs': 100,
            'num_workers': 4,
            'pin_memory': True,
            'grad_clip': 1.0
        },
        'optimizer': {
            'type': 'adamw',
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'betas': [0.9, 0.999]
        },
        'scheduler': {
            'enabled': True,
            'type': 'cosine',
            'eta_min': 0.000001
        },
        'loss': {
            'type': 'fog_density_aware',
            'base_loss': 'cross_entropy',
            'depth_weight': 0.5,
            'fog_sensitivity': 2.0,
            'depth_loss_weight': 0.1
        },
        'early_stopping': {
            'patience': 10,
            'min_delta': 0.001,
            'restore_best_weights': True
        },
        'mlflow': {
            'enabled': True,
            'experiment_name': 'adverse_weather_segmentation',
            'run_name': None
        },
        'evaluation': {
            'num_bins': 15,
            'weather_conditions': ['clean', 'fog', 'rain', 'snow', 'night']
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'paths': {
            'checkpoints': 'checkpoints',
            'logs': 'logs',
            'results': 'results'
        },
        'device': 'auto',  # 'auto', 'cpu', 'cuda', 'cuda:0', etc.
        'seed': 42
    }

    return Config(default_config)


def get_device_config(device_setting: str = 'auto') -> str:
    """
    Get appropriate device configuration.

    Args:
        device_setting: Device setting ('auto', 'cpu', 'cuda', etc.)

    Returns:
        Device string for PyTorch
    """
    if device_setting == 'auto':
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        except ImportError:
            logger.warning("PyTorch not available, defaulting to CPU")
            return 'cpu'
    else:
        return device_setting


def setup_logging(config: Config) -> None:
    """
    Setup logging configuration.

    Args:
        config: Configuration object
    """
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        force=True
    )

    logger.info("Logging configured")


def validate_config(config: Config) -> None:
    """
    Validate configuration for required fields and valid values.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = [
        'model.num_classes',
        'data.image_size',
        'training.batch_size',
        'training.epochs',
        'optimizer.learning_rate'
    ]

    for field in required_fields:
        if config.get(field) is None:
            raise ValueError(f"Required configuration field missing: {field}")

    # Validate specific values
    if config.get('model.num_classes', 0) <= 0:
        raise ValueError("model.num_classes must be positive")

    if config.get('training.batch_size', 0) <= 0:
        raise ValueError("training.batch_size must be positive")

    if config.get('training.epochs', 0) <= 0:
        raise ValueError("training.epochs must be positive")

    if config.get('optimizer.learning_rate', 0) <= 0:
        raise ValueError("optimizer.learning_rate must be positive")

    # Validate image size
    image_size = config.get('data.image_size')
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise ValueError("data.image_size must be a list of two integers [height, width]")

    logger.info("Configuration validation passed")