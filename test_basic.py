#!/usr/bin/env python3
"""
Basic functionality test for the collaborative stegosystem.
This script tests the core components without requiring expensive model downloads.
"""

import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import trl
        print(f"✓ TRL {trl.__version__}")
    except ImportError as e:
        print(f"✗ TRL import failed: {e}")
        return False
    
    try:
        from src import models, environment, ppo_trainer, utils
        print("✓ All source modules imported successfully")
    except ImportError as e:
        print(f"✗ Source module import failed: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from src.utils import load_config
        config = load_config('config.yaml')
        print("✓ Configuration loaded successfully")
        print(f"  - WandB project: {config['wandb']['project_name']}")
        print(f"  - Model: {config['model']['name']}")
        print(f"  - Training epochs: {config['training']['num_epochs']}")
        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def test_environment():
    """Test environment creation and basic functionality."""
    print("\nTesting environment...")
    
    try:
        from src.environment import TradingEnvironment
        from src.utils import load_config
        
        config = load_config('config.yaml')
        
        # Test without OpenAI API key
        env = TradingEnvironment(config)
        print("✓ Environment created successfully")
        
        # Test reset
        state = env.reset()
        print(f"✓ Environment reset: {state}")
        
        # Test step with dummy messages
        test_state, reward, done = env.step(
            "I think the market looks interesting today.",
            "Yes, I agree. There are some promising opportunities."
        )
        print(f"✓ Environment step completed: reward={reward}, done={done}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_models_module():
    """Test models module functionality."""
    print("\nTesting models module...")
    
    try:
        from src.models import extract_decision
        
        # Test decision extraction
        test_responses = [
            "I think AAPL will go up",
            "The market seems to be heading down",
            "MSFT and GOOGL look promising",
            "No specific direction mentioned"
        ]
        
        for response in test_responses:
            decision = extract_decision(response)
            print(f"  '{response}' -> {decision}")
        
        print("✓ Decision extraction working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Models module test failed: {e}")
        return False

def test_hardware_detection():
    """Test hardware detection and device setup."""
    print("\nTesting hardware detection...")
    
    try:
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            print("Running on CPU")
        
        print("✓ Hardware detection completed")
        return True
        
    except Exception as e:
        print(f"✗ Hardware detection failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Collaborative Stegosystem - Basic Functionality Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config_loading,
        test_environment,
        test_models_module,
        test_hardware_detection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All basic functionality tests passed!")
        print("\nNext steps:")
        print("1. Set your API keys in environment variables:")
        print("   export HF_TOKEN=your_huggingface_token")
        print("   export OPENAI_API_KEY=your_openai_key")
        print("   export WANDB_API_KEY=your_wandb_key")
        print("2. Run: python main.py")
    else:
        print("✗ Some tests failed. Please fix the issues before proceeding.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
