#!/usr/bin/env python3
"""
Test script to simulate training functionality without deep learning dependencies.
This validates that the training pipeline structure is correct.
"""

import sys
import os
import ast
import importlib.util
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_config_module():
    """Test configuration module functionality."""
    print("Testing configuration module...")

    try:
        from adverse_weather_semantic_segmentation_robustness_benchmark.utils.config import (
            Config, create_default_config, validate_config
        )

        # Test default config creation
        config = create_default_config()
        print("✓ Default config created successfully")

        # Test config access
        num_classes = config.get('model.num_classes')
        assert num_classes == 19, f"Expected 19 classes, got {num_classes}"
        print(f"✓ Config access works (num_classes: {num_classes})")

        # Test config validation
        validate_config(config)
        print("✓ Config validation passed")

        # Test config modification
        config.set('model.num_classes', 21)
        assert config.get('model.num_classes') == 21
        print("✓ Config modification works")

        return True

    except Exception as e:
        print(f"✗ Configuration module test failed: {e}")
        return False

def test_script_structure():
    """Test that training scripts have correct structure."""
    print("\nTesting script structure...")

    scripts = {
        'train.py': 'scripts/train.py',
        'evaluate.py': 'scripts/evaluate.py'
    }

    for script_name, script_path in scripts.items():
        try:
            # Read and parse the script
            with open(script_path, 'r') as f:
                content = f.read()

            # Parse AST to verify structure
            tree = ast.parse(content)

            # Check for main function
            has_main = any(
                isinstance(node, ast.FunctionDef) and node.name == 'main'
                for node in ast.walk(tree)
            )

            if has_main:
                print(f"✓ {script_name} has main function")
            else:
                print(f"✗ {script_name} missing main function")
                return False

            # Check for argparse usage
            has_argparse = 'argparse' in content
            if has_argparse:
                print(f"✓ {script_name} uses argparse")
            else:
                print(f"? {script_name} may not use argparse")

        except Exception as e:
            print(f"✗ Error analyzing {script_name}: {e}")
            return False

    return True

def test_module_structure():
    """Test that all modules can be imported without dependencies."""
    print("\nTesting module structure...")

    # Test imports that don't require heavy dependencies
    modules_to_test = [
        'adverse_weather_semantic_segmentation_robustness_benchmark',
        'adverse_weather_semantic_segmentation_robustness_benchmark.utils',
        'adverse_weather_semantic_segmentation_robustness_benchmark.utils.config',
    ]

    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            print(f"✓ Successfully imported {module_name}")
        except ImportError as e:
            if 'torch' in str(e) or 'transformers' in str(e) or 'cv2' in str(e):
                print(f"? {module_name} requires deep learning dependencies (expected)")
            else:
                print(f"✗ Unexpected import error in {module_name}: {e}")
                return False
        except Exception as e:
            print(f"✗ Error importing {module_name}: {e}")
            return False

    return True

def test_training_script_args():
    """Test that training script accepts expected arguments."""
    print("\nTesting training script argument parsing...")

    try:
        # Read train.py and check for argument definitions
        with open('scripts/train.py', 'r') as f:
            content = f.read()

        expected_args = ['--config', '--resume', '--device', '--seed', '--output-dir']

        for arg in expected_args:
            if arg in content:
                print(f"✓ Found argument: {arg}")
            else:
                print(f"✗ Missing argument: {arg}")
                return False

        return True

    except Exception as e:
        print(f"✗ Error testing training script args: {e}")
        return False

def test_checkpoint_and_logging_structure():
    """Test that checkpoint and logging structure is properly implemented."""
    print("\nTesting checkpoint and logging structure...")

    try:
        # Read trainer module to check for checkpoint methods
        with open('src/adverse_weather_semantic_segmentation_robustness_benchmark/training/trainer.py', 'r') as f:
            trainer_content = f.read()

        # Check for essential methods
        essential_methods = [
            'save_checkpoint',
            'load_checkpoint',
            'train_epoch',
            'validate_epoch'
        ]

        for method in essential_methods:
            if f"def {method}" in trainer_content:
                print(f"✓ Found trainer method: {method}")
            else:
                print(f"✗ Missing trainer method: {method}")
                return False

        # Check for logging setup
        if 'logging' in trainer_content and 'logger' in trainer_content:
            print("✓ Logging infrastructure present")
        else:
            print("✗ Missing logging infrastructure")
            return False

        return True

    except Exception as e:
        print(f"✗ Error testing checkpoint/logging structure: {e}")
        return False

def test_requirements_completeness():
    """Test that requirements.txt includes all necessary packages."""
    print("\nTesting requirements completeness...")

    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().lower()

        # Essential packages for training
        essential_packages = [
            'torch', 'torchvision', 'numpy', 'opencv-python',
            'transformers', 'segmentation-models-pytorch',
            'tensorboard', 'mlflow', 'albumentations', 'tqdm'
        ]

        missing_packages = []
        for package in essential_packages:
            if package not in requirements:
                missing_packages.append(package)
                print(f"✗ Missing package: {package}")
            else:
                print(f"✓ Found package: {package}")

        if missing_packages:
            print(f"Missing {len(missing_packages)} essential packages")
            return False

        return True

    except Exception as e:
        print(f"✗ Error testing requirements: {e}")
        return False

def main():
    """Run all validation tests."""
    print("Running training pipeline validation tests...\n")

    tests = [
        ("Configuration Module", test_config_module),
        ("Script Structure", test_script_structure),
        ("Module Structure", test_module_structure),
        ("Training Script Arguments", test_training_script_args),
        ("Checkpoint and Logging", test_checkpoint_and_logging_structure),
        ("Requirements Completeness", test_requirements_completeness),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"{'=' * 50}")
        print(f"Running {test_name} Test")
        print(f"{'=' * 50}")

        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'=' * 50}")
    print("TRAINING VALIDATION SUMMARY")
    print(f"{'=' * 50}")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All training validation tests passed!")
        print("The training pipeline structure is correct and should work with dependencies installed.")
        return True
    else:
        print("❌ Some validation tests failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)