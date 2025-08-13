# main.py
import wandb
from src.environment import TradingEnvironment
from src.ppo_trainer import CollaborativePPOTrainer
from src.utils import load_config, init_wandb
import torch

import warnings
warnings.filterwarnings("ignore", message="No dataset is provided.")
warnings.filterwarnings("ignore", message="Detected kernel version")

def main():
    config = load_config('config.yaml')
    
    init_wandb(config)
    
    wandb.log({
        "system/gpu_count": torch.cuda.device_count(),
        "system/gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "system/cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
    })
    
    env = TradingEnvironment(config)
    trainer = CollaborativePPOTrainer(config)
    trainer.train(env, num_epochs=config['training']['num_epochs'])
    
    wandb.finish()

if __name__ == "__main__":
    main()