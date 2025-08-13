import yaml
import os
import wandb

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
def init_wandb(config):
    project = config.get('wandb', {}).get('project_name', 'collaborative-stegosystem')
    entity = config.get('wandb', {}).get('entity')
    # Auto-disable if no API key
    default_mode = 'disabled' if not os.environ.get('WANDB_API_KEY') else 'online'
    mode = config.get('wandb', {}).get('mode', os.environ.get('WANDB_MODE', default_mode))
    wandb.init(project=project, entity=entity, config=config, mode=mode)