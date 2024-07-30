import argparse
from utils.config_loader import load_config
from experiment import Experiment

def main(config_path: str):
    config = load_config(config_path)
    experiment = Experiment(config)
    experiment.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stock_market_config.yaml")
    args = parser.parse_args()
    
    main(args.config)