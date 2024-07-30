import yaml
import os

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Expand user for paths
    if 'stock_data_path' in config:
        config['stock_data_path'] = os.path.expanduser(config['stock_data_path'])
    if 'results_dir' in config:
        config['results_dir'] = os.path.expanduser(config['results_dir'])
    
    return config