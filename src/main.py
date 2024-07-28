import argparse
from utils.config_loader import load_config
from agents.llm_agent import LLMAgent
from experiment import Experiment
import importlib

def main(config_path, environment_name):
    config = load_config(config_path)

    # Dynamically import the specified environment
    env_module = importlib.import_module(f"environments.{environment_name}")
    EnvironmentClass = getattr(env_module, environment_name.capitalize() + "Environment")
    
    environment = EnvironmentClass(config)

    alice = LLMAgent("Alice", config['alice_secret'], config['model_name'])
    bob = LLMAgent("Bob", config['bob_secret'], config['model_name'])
    eve = LLMAgent("Eve", "I am the overseer", config['model_name'])

    experiment = Experiment(config, environment, alice, bob, eve)
    results = experiment.run()

    print(f"Experiment completed with {len(results)} total steps across all episodes.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--environment", type=str, default="simple")
    args = parser.parse_args()
    
    main(args.config, args.environment)