import wandb
from src.environment import TradingEnvironment
from src.ppo_trainer import CollaborativePPOTrainer
from src.utils import load_config
import torch

import warnings
warnings.filterwarnings("ignore", message="No dataset is provided.")
warnings.filterwarnings("ignore", message="Detected kernel version")


def main():
    config = load_config('config.yaml')
    
    wandb.init(project=config['wandb']['project_name'], entity=config['wandb']['entity'], config=config)
    
    # Log system information
    wandb.log({
        "system/gpu_count": torch.cuda.device_count(),
        "system/gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "system/cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
    })
    
    env = TradingEnvironment(config)
    trainer = CollaborativePPOTrainer(config)
    trainer.train(env)
    
    wandb.finish()

if __name__ == "__main__":
    main()