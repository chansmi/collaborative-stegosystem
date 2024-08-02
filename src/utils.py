import yaml
import wandb

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
def init_wandb(config):
    wandb.init(project=config['experiment']['wandb_project'], entity=config['experiment']['wandb_entity'], config=config)