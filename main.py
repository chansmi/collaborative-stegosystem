# main.py
from src.environment import TradingEnvironment
from src.ppo_trainer import CollaborativePPOTrainer
from src.utils import load_config

def main():
    try:
        config = load_config('config.yaml')
        env = TradingEnvironment(config)
        trainer = CollaborativePPOTrainer(config)
        trainer.train(env)
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    main()