import os
import argparse
import torch
import torch.distributed as dist
from src.environment import TradingEnvironment
from src.ppo_trainer import CollaborativePPOTrainer, DistributedWrapper
from src.utils import load_config

def setup():
    rank = int(os.environ.get('FLUX_TASK_RANK', 0))
    world_size = int(os.environ.get('FLUX_JOB_SIZE', 1))
    if world_size > 1:
        os.environ['MASTER_ADDR'] = os.environ.get('FLUX_JOB_BROKER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('FLUX_JOB_BROKER_PORT', '12345')
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Run Collaborative Stegosystem training")
    parser.add_argument("--distributed", action="store_true", help="Run in distributed mode")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    env = TradingEnvironment(config)

    if args.distributed:
        rank = int(os.environ.get('FLUX_TASK_RANK', 0))
        world_size = int(os.environ.get('FLUX_JOB_SIZE', 1))
        trainer = DistributedWrapper(CollaborativePPOTrainer, config, rank, world_size)
    else:
        trainer = CollaborativePPOTrainer(config)

    trainer.train(env)

if __name__ == "__main__":
    main()