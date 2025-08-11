#!/usr/bin/env python3
"""
Test script for the main training configuration.
Tests the full system with Llama-2 but with fewer epochs.
"""

import os
import sys
import wandb
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_main_training():
    """Test the main training configuration with fewer epochs."""
    print("Testing main training configuration...")
    
    try:
        from src.environment import TradingEnvironment
        from src.ppo_trainer import CollaborativePPOTrainer
        from src.utils import load_config
        
        # Load main configuration
        config = load_config('config.yaml')
        print("✓ Main configuration loaded")
        
        # Modify config for testing
        config['training']['num_epochs'] = 3  # Just 3 epochs for testing
        config['model']['quantize'] = False   # Disable quantization for testing
        
        print(f"  - Model: {config['model']['name']}")
        print(f"  - Training epochs: {config['training']['num_epochs']}")
        print(f"  - Quantization: {config['model']['quantize']}")
        
        # Initialize wandb
        try:
            wandb.init(
                project=config['wandb']['project_name'], 
                entity=config['wandb']['entity'], 
                config=config,
                mode="disabled"  # Disable wandb for testing
            )
            print("✓ WandB initialized (disabled mode)")
        except Exception as e:
            print(f"Warning: WandB initialization failed: {e}")
            print("Continuing without WandB...")
        
        # Create environment
        env = TradingEnvironment(config)
        print("✓ Environment created")
        
        # Create trainer
        trainer = CollaborativePPOTrainer(config)
        print("✓ Trainer created")
        
        # Run training
        print("\nStarting main training test...")
        trainer.train(env, num_epochs=config['training']['num_epochs'])
        
        print("✓ Main training test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Main training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the main training test."""
    print("=" * 60)
    print("Collaborative Stegosystem - Main Training Test")
    print("=" * 60)
    
    # Check environment variables
    required_vars = ['HF_TOKEN', 'OPENAI_API_KEY', 'WANDB_API_KEY']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print("Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        return 1
    
    print("Environment variables check passed ✓")
    
    # Run the test
    if test_main_training():
        print("\n" + "=" * 60)
        print("✓ Main training test completed successfully!")
        print("The full collaborative stegosystem is working correctly.")
        print("\nYou can now:")
        print("1. Run full training: python main.py")
        print("2. Experiment with different configurations")
        print("3. Deploy to GPU-enabled distributed computing")
        return 0
    else:
        print("\n" + "=" * 60)
        print("✗ Main training test failed!")
        print("Please check the error messages above and fix any issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
