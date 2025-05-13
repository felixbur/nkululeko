"""
Test script for verifying nkululeko installation.
This script creates a virtual environment, installs nkululeko,
and runs module tests to ensure the installation works correctly.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    """Run a command and return its output"""
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return process.stdout, process.stderr, process.returncode


def main(python_version=None):
    """
    Test nkululeko installation in a virtual environment.

    Args:
        python_version: Optional Python version to use (e.g., "3.10", "3.11", "3.12")
    """
    repo_root = Path(__file__).resolve().parent.parent

    python_cmd = f"python{python_version}" if python_version else "python"

    # check python version
    stdout, stderr, returncode = run_command(f"{python_cmd} --version")
    print(f"Using Python: {python_cmd}")
    print(f"Python version: {stdout.strip()}")
    python_ver = stdout.strip().split()[1]  # Extract version number from "Python X.Y.Z"

    # create test directory
    test_dir = repo_root / "build"
    print(f"Creating test directory: {test_dir}")

    # check if the directory exists, give message to reuse
    if test_dir.exists():
        print(
            f"Test directory {test_dir}/{python_ver} exists and will be reused. "
            "If you want to start fresh, please delete it first."
        )
    os.makedirs(test_dir, exist_ok=True)
    os.chdir(test_dir)

    print("Creating virtual environment...")
    stdout, stderr, returncode = run_command(f"{python_cmd} -m venv {python_ver}")
    if returncode != 0:
        print(f"Failed to create virtual environment ({python_ver} exists): {stderr}")
        return 1

    pip_cmd = f"./{python_ver}/bin/pip"

    venv_python = f"./{python_ver}/bin/python"
    print("Installing nkululeko...")
    stdout, stderr, returncode = run_command(f"{pip_cmd} install -e {repo_root}")
    if returncode != 0:
        print(f"Failed to install nkululeko: {stderr}")
        return 1

    print("Basic installation successful")

    with open("test_import.py", "w") as f:
        f.write(
            "import nkululeko\nprint(f'Nkululeko version: {nkululeko.__version__}')"
        )

    print("Testing import...")
    stdout, stderr, returncode = run_command(f"{venv_python} test_import.py")
    if returncode != 0:
        print(f"Failed to import nkululeko: {stderr}")
        return 1

    print(f"Import test output: {stdout.strip()}")

    # ensure python version is 3.12 or lower
    if python_ver >= "3.13":
        print(
            f"Python version {python_ver} is not supported for spotlight dependencies."
        )
        # skip spotlight installation

    else:
        print("Installing spotlight dependencies...")
        stdout, stderr, returncode = run_command(
            f'{pip_cmd} install "renumics-spotlight>=1.6.13" "sliceguard>=0.0.35"'
        )
        if returncode != 0:
            print(f"Failed to install spotlight dependencies: {stderr}")
            return 1
        print("Spotlight dependencies installed successfully")

    print("Installing torch dependencies for tests...")
    stdout, stderr, returncode = run_command(
        f'{pip_cmd} install "torch>=1.0.0" "torchvision>=0.10.0" "torchaudio>=0.10.0"'
    )
    if returncode != 0:
        print(f"Failed to install torch dependencies: {stderr}")

    with open("run_tests.py", "w") as f:
        f.write(
            f"""
import unittest
import sys
import os
sys.path.insert(0, os.path.expanduser('{repo_root}'))
from tests.test_modules import TestModules

if __name__ == '__main__':
    unittest.main()
"""
        )

    print("Running unit tests...")
    stdout, stderr, returncode = run_command(f"{venv_python} run_tests.py")
    if returncode != 0:
        print(f"Unit tests failed: {stderr}")
        return 1

    print("Unit tests passed")
    print("All tests completed successfully!")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test nkululeko installation")
    parser.add_argument(
        "--python", help="Python version to use (e.g., 3.10, 3.11, 3.12)"
    )
    args = parser.parse_args()

    sys.exit(main(args.python))
