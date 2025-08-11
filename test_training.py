#!/usr/bin/env python3
"""
Test training script for the collaborative stegosystem.
This script tests the training pipeline with a small model.
"""

import os
import sys
import wandb
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_training():
    """Test the training pipeline with a small model."""
    print("Testing training pipeline...")
    
    try:
        from src.environment import TradingEnvironment
        from src.ppo_trainer import CollaborativePPOTrainer
        from src.utils import load_config
        
        # Load test configuration
        config = load_config('config_test.yaml')
        print("✓ Test configuration loaded")
        
        # Initialize wandb (optional for testing)
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
        
        # Run a few training steps
        print("\nStarting test training...")
        trainer.train(env, num_epochs=config['training']['num_epochs'])
        
        print("✓ Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the training test."""
    print("=" * 60)
    print("Collaborative Stegosystem - Training Test")
    print("=" * 60)
    
    # Check if we have the required environment variables
    required_vars = ['HF_TOKEN']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print("Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set them before running:")
        print("  export HF_TOKEN=your_huggingface_token")
        return 1
    
    print("Environment variables check passed ✓")
    
    # Run the test
    if test_training():
        print("\n" + "=" * 60)
        print("✓ Training test completed successfully!")
        print("The collaborative stegosystem is working correctly.")
        print("\nYou can now:")
        print("1. Use the full configuration with Llama-2")
        print("2. Run longer training sessions")
        print("3. Experiment with different parameters")
        return 0
    else:
        print("\n" + "=" * 60)
        print("✗ Training test failed!")
        print("Please check the error messages above and fix any issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
