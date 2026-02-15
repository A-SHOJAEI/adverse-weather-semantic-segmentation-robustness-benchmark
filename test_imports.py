#!/usr/bin/env python3
"""
Minimal test script to validate project structure and imports without external dependencies.
This script checks if the code is well-structured and can be imported properly.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test basic package imports."""
    print("Testing package structure...")

    try:
        # Test package init
        import adverse_weather_semantic_segmentation_robustness_benchmark
        print("✓ Package import successful")
    except Exception as e:
        print(f"✗ Package import failed: {e}")
        return False

    try:
        # Test submodule imports without dependencies
        from adverse_weather_semantic_segmentation_robustness_benchmark import utils
        print("✓ Utils module import successful")
    except Exception as e:
        print(f"✗ Utils module import failed: {e}")
        return False

    return True

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")

    required_files = [
        "src/adverse_weather_semantic_segmentation_robustness_benchmark/__init__.py",
        "src/adverse_weather_semantic_segmentation_robustness_benchmark/models/__init__.py",
        "src/adverse_weather_semantic_segmentation_robustness_benchmark/models/model.py",
        "src/adverse_weather_semantic_segmentation_robustness_benchmark/data/__init__.py",
        "src/adverse_weather_semantic_segmentation_robustness_benchmark/data/loader.py",
        "src/adverse_weather_semantic_segmentation_robustness_benchmark/data/preprocessing.py",
        "src/adverse_weather_semantic_segmentation_robustness_benchmark/training/__init__.py",
        "src/adverse_weather_semantic_segmentation_robustness_benchmark/training/trainer.py",
        "src/adverse_weather_semantic_segmentation_robustness_benchmark/evaluation/__init__.py",
        "src/adverse_weather_semantic_segmentation_robustness_benchmark/evaluation/metrics.py",
        "src/adverse_weather_semantic_segmentation_robustness_benchmark/utils/__init__.py",
        "src/adverse_weather_semantic_segmentation_robustness_benchmark/utils/config.py",
        "scripts/train.py",
        "scripts/evaluate.py",
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/test_data.py",
        "tests/test_model.py",
        "tests/test_training.py",
        "requirements.txt",
        "README.md"
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"✗ Missing: {file_path}")
        else:
            print(f"✓ Found: {file_path}")

    if missing_files:
        print(f"\n{len(missing_files)} files are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False

    print(f"\n✓ All {len(required_files)} required files found")
    return True

def test_python_syntax():
    """Test that all Python files have valid syntax."""
    print("\nTesting Python syntax...")

    python_files = list(Path(".").rglob("*.py"))
    syntax_errors = []

    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                compile(f.read(), py_file, 'exec')
            print(f"✓ {py_file}")
        except SyntaxError as e:
            syntax_errors.append((py_file, e))
            print(f"✗ {py_file}: {e}")
        except Exception as e:
            print(f"? {py_file}: Could not read file - {e}")

    if syntax_errors:
        print(f"\n{len(syntax_errors)} files have syntax errors:")
        for file_path, error in syntax_errors:
            print(f"  - {file_path}: {error}")
        return False

    print(f"\n✓ All {len(python_files)} Python files have valid syntax")
    return True

def test_requirements():
    """Test that requirements.txt contains essential dependencies."""
    print("\nTesting requirements.txt...")

    try:
        with open("requirements.txt", 'r') as f:
            requirements = f.read()

        essential_deps = [
            "torch", "torchvision", "numpy", "opencv-python", "albumentations",
            "matplotlib", "scipy", "scikit-learn", "pytest", "pyyaml",
            "transformers", "segmentation-models-pytorch"
        ]

        missing_deps = []
        for dep in essential_deps:
            if dep not in requirements:
                missing_deps.append(dep)
                print(f"✗ Missing dependency: {dep}")
            else:
                print(f"✓ Found dependency: {dep}")

        if missing_deps:
            print(f"\n{len(missing_deps)} essential dependencies are missing")
            return False

        print(f"\n✓ All {len(essential_deps)} essential dependencies found")
        return True

    except FileNotFoundError:
        print("✗ requirements.txt not found")
        return False

def main():
    """Run all tests."""
    print("Running project validation tests...\n")

    tests = [
        ("File Structure", test_file_structure),
        ("Python Syntax", test_python_syntax),
        ("Requirements", test_requirements),
        ("Basic Imports", test_imports)
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
    print("TEST SUMMARY")
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
        print("🎉 All tests passed! The project structure is valid.")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)