import wandb
from tqdm import tqdm
from src.environment import TradingEnvironment
from src.models import create_agent
from src.ppo_trainer import CollaborativePPOTrainer
from src.utils import load_config, init_wandb
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    try:
        config = load_config()
        wandb.init(project=config['experiment']['wandb_project'], 
                   entity=config['experiment']['wandb_entity'], 
                   config=config)


        env = TradingEnvironment(config)
        
        alice = create_agent(config, "Alice")
        bob = create_agent(config, "Bob")
        eve = create_agent(config, "Eve")

        trainer = CollaborativePPOTrainer(config, alice, bob, eve, env)

        progress_bar = tqdm(range(config['training']['num_epochs']), desc="Training Progress")
        for epoch in progress_bar:
            total_reward, _ = trainer.train_step()
            progress_bar.set_postfix({"Reward": f"{total_reward:.2f}"})

        wandb.finish()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        wandb.finish()

if __name__ == "__main__":
    main()