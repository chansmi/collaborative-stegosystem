#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Approach for Collaborative Steganography

This is the standard approach using the TRL library's PPO implementation.
It provides a solid baseline for training agents to communicate secretly.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def run_ppo_training(config_path="config_gpu.yaml", num_epochs=10):
    """
    Run PPO training with the specified configuration.
    
    Args:
        config_path (str): Path to configuration file
        num_epochs (int): Number of training epochs
    """
    try:
        from src.environment import TradingEnvironment
        from src.ppo_trainer import CollaborativePPOTrainer
        from src.utils import load_config
        
        print("🚀 Starting PPO Training for Collaborative Steganography")
        print("=" * 60)
        
        # Load configuration
        config = load_config(config_path)
        config['training']['num_epochs'] = num_epochs
        
        print(f"📋 Configuration: {config_path}")
        print(f"🎯 Model: {config['model']['name']}")
        print(f"⏱️  Epochs: {num_epochs}")
        print(f"🔧 Quantization: {config['model'].get('quantize', False)}")
        
        # Create environment
        print("\n🏗️  Creating trading environment...")
        env = TradingEnvironment(config)
        print("✅ Environment created successfully")
        
        # Create trainer
        print("\n🤖 Creating PPO trainer...")
        trainer = CollaborativePPOTrainer(config)
        print("✅ Trainer created successfully")
        
        # Start training
        print(f"\n🎓 Starting training for {num_epochs} epochs...")
        print("=" * 60)
        
        trainer.train(env, num_epochs=num_epochs)
        
        print("\n🎉 Training completed successfully!")
        print("📊 Check the results directory for saved models and logs")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_test():
    """Run a quick test with minimal epochs."""
    print("🧪 Running Quick PPO Test")
    return run_ppo_training(num_epochs=3)

def run_full_training():
    """Run full training with default epochs."""
    print("🏃 Running Full PPO Training")
    return run_ppo_training(num_epochs=100)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PPO Training for Collaborative Steganography")
    parser.add_argument("--config", default="config_gpu.yaml", help="Configuration file path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--quick", action="store_true", help="Run quick test (3 epochs)")
    parser.add_argument("--full", action="store_true", help="Run full training (100 epochs)")
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_test()
    elif args.full:
        success = run_full_training()
    else:
        success = run_ppo_training(args.config, args.epochs)
    
    sys.exit(0 if success else 1)
