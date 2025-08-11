#!/usr/bin/env python3
"""
Setup script for the collaborative stegosystem.
Installs dependencies and sets up the environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e}")
        if e.stdout:
            print(f"  stdout: {e.stdout}")
        if e.stderr:
            print(f"  stderr: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"✗ Python {version.major}.{version.minor} is not supported.")
        print("Please use Python 3.10 or later.")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    
    # Upgrade pip first
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    
    directories = [
        "models",
        "results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def check_environment():
    """Check environment variables."""
    print("\nChecking environment variables...")
    
    required_vars = ['HF_TOKEN']
    optional_vars = ['OPENAI_API_KEY', 'WANDB_API_KEY']
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_required.append(var)
    
    for var in optional_vars:
        if not os.environ.get(var):
            missing_optional.append(var)
    
    if missing_required:
        print("✗ Missing required environment variables:")
        for var in missing_required:
            print(f"  - {var}")
        print("\nPlease set them:")
        print("  export HF_TOKEN=your_huggingface_token")
        return False
    
    print("✓ Required environment variables are set")
    
    if missing_optional:
        print("⚠ Missing optional environment variables:")
        for var in missing_optional:
            print(f"  - {var}")
        print("These are not required for basic functionality but enable:")
        print("  - OPENAI_API_KEY: Eve's detection system")
        print("  - WANDB_API_KEY: Experiment tracking")
    
    return True

def run_tests():
    """Run basic tests to verify setup."""
    print("\nRunning basic tests...")
    
    if not run_command("python test_basic.py", "Basic functionality tests"):
        print("⚠ Basic tests failed, but setup may still work")
        return True
    
    return True

def main():
    """Main setup function."""
    print("=" * 60)
    print("Collaborative Stegosystem - Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("\n✗ Setup failed during dependency installation")
        return 1
    
    # Create directories
    if not create_directories():
        print("\n✗ Setup failed during directory creation")
        return 1
    
    # Check environment
    if not check_environment():
        print("\n✗ Setup failed during environment check")
        return 1
    
    # Run tests
    if not run_tests():
        print("\n⚠ Setup completed but tests failed")
        print("You may still be able to use the system")
    
    print("\n" + "=" * 60)
    print("✓ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Set your API keys if you haven't already:")
    print("   export HF_TOKEN=your_huggingface_token")
    print("   export OPENAI_API_KEY=your_openai_key (optional)")
    print("   export WANDB_API_KEY=your_wandb_key (optional)")
    print("\n2. Test the system:")
    print("   python test_basic.py")
    print("   python test_training.py")
    print("\n3. Run full training:")
    print("   python main.py")
    print("\n4. Check the README.md for more information")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
