# main.py
import os
import wandb
from src.environment import TradingEnvironment
from src.ppo_trainer import CollaborativePPOTrainer
from src.utils import load_config

def main():
    try:
        # Ensure the OpenAI API key is set as an environment variable
        if "OPENAI_API_KEY" not in os.environ:
            print("Error: OPENAI_API_KEY environment variable is not set.")
            print("Please set it by running: export OPENAI_API_KEY='your-api-key-here'")
            return


        config = load_config('config.yaml')
        env = TradingEnvironment(config)
        trainer = CollaborativePPOTrainer(config)
        trainer.train(env)
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()