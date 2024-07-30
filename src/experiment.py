import os
import json
from datetime import datetime
from typing import Dict
import wandb
from src.environments.stock_market_environment import StockMarketEnvironment
from src.agents.alice import Alice
from src.agents.bob import Bob
from src.agents.eve import Eve
from src.trainers.ppo_trainer import PPOTrainer

class Experiment:
    def __init__(self, config: Dict):
        self.config = config
        self.env = StockMarketEnvironment(config)
        
        state_dim = len(self.env.companies) * 5 + 1  # 5 features per company + cash
        action_dim = len(self.env.companies)
        
        self.alice = Alice(state_dim, action_dim, "Alice")
        self.bob = Bob(state_dim, action_dim, "Bob")
        self.eve = Eve()
        
        self.trainer = PPOTrainer(self.env, self.alice, self.bob, self.eve, config)
        
        self.results_dir = config['results_dir']
        os.makedirs(self.results_dir, exist_ok=True)

    def run(self):
        wandb.init(project="Collaborative Stegosystem", entity="chansmi", config=self.config)

        self.trainer.train(self.config['num_episodes'])
        
        self.save_results()
        
        wandb.finish()

    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        results = {
            "config": self.config,
            "final_portfolio_value": self.env.get_state()['cash'],
            "alice_portfolio": self.env.alice_portfolio,
            "bob_portfolio": self.env.bob_portfolio
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")