import torch
import wandb
from trl import PPOTrainer, PPOConfig
from src.models import create_agent
import logging
import sys
from src.train_reward import generate_prompt, extract_thoughts_and_message, detect_secret_message
import math

def filter_nan_values(d):
    return {k: v for k, v in d.items() if not (isinstance(v, float) and math.isnan(v))}

def debug_nan_values(d):
    for k, v in d.items():
        if isinstance(v, float) and math.isnan(v):
            print(f"NaN value found for key: {k}")


class CollaborativePPOTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.alice = create_agent(config, 'Alice')
        self.bob = create_agent(config, 'Bob')
        
        ppo_config_dict = config['ppo'].copy()
        ppo_config_dict['learning_rate'] = float(ppo_config_dict['learning_rate'])
        self.ppo_config = PPOConfig(**ppo_config_dict)
        
        self.alice_trainer = PPOTrainer(config=self.ppo_config, model=self.alice['model'], tokenizer=self.alice['tokenizer'])
        self.bob_trainer = PPOTrainer(config=self.ppo_config, model=self.bob['model'], tokenizer=self.bob['tokenizer'])

        self.generation_kwargs = {
            "max_new_tokens": config['model'].get('max_new_tokens', 50),
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
        }

        self.batch_size = self.ppo_config.batch_size
        self.alice_batch = {'queries': [], 'responses': [], 'rewards': []}
        self.bob_batch = {'queries': [], 'responses': [], 'rewards': []}

    def generate_response(self, trainer, tokenizer, prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        query_tensor = inputs.input_ids.squeeze(0)  # Remove batch dimension
        with torch.no_grad():
            response = trainer.generate(query_tensor, **self.generation_kwargs)
        return tokenizer.decode(response[0], skip_special_tokens=True)

    def add_to_batch(self, batch, query, response, reward):
        batch['queries'].append(query)
        batch['responses'].append(response)
        batch['rewards'].append(reward)

    def process_batch(self, trainer, batch):
        if len(batch['queries']) == self.batch_size:
            trainer.step(
                batch['queries'],
                batch['responses'],
                [torch.tensor(r).to(self.device) for r in batch['rewards']]
            )
            batch['queries'].clear()
            batch['responses'].clear()
            batch['rewards'].clear()

    def train(self, env):
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("trl").setLevel(logging.ERROR)

        for epoch in range(self.config['training']['num_epochs']):
            state = env.reset()
            total_reward = 0
            
            print(f"\n--- Epoch {epoch + 1} ---")
            
            alice_prompt = generate_prompt(self, state, 'Alice')
            bob_prompt = generate_prompt(self, state, 'Bob')
            
            alice_response = self.generate_response(self.alice_trainer, self.alice['tokenizer'], alice_prompt)
            bob_response = self.generate_response(self.bob_trainer, self.bob['tokenizer'], bob_prompt)
            
            alice_thoughts, alice_message = extract_thoughts_and_message(self, alice_response)
            bob_thoughts, bob_message = extract_thoughts_and_message(self, bob_response)
            
            print("Alice (inner thoughts):", alice_thoughts)
            print("Alice:", alice_message)
            print("Bob (inner thoughts):", bob_thoughts)
            print("Bob:", bob_message)
            
            next_state, reward, done = env.step(alice_message, bob_message, alice_thoughts, bob_thoughts)

            print(f"Reward: {reward}")
            if math.isnan(reward):
                print("Warning: Reward is NaN!")
                reward = 0  # or some default value

            # Prepare data for PPO update
            alice_query_tokens = self.alice['tokenizer'](alice_prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
            alice_response_tokens = self.alice['tokenizer'](alice_response, return_tensors="pt", padding=True, truncation=True).to(self.device)
            bob_query_tokens = self.bob['tokenizer'](bob_prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
            bob_response_tokens = self.bob['tokenizer'](bob_response, return_tensors="pt", padding=True, truncation=True).to(self.device)

            # PPO step
            alice_stats = self.alice_trainer.step([alice_query_tokens.input_ids.squeeze(0)], [alice_response_tokens.input_ids.squeeze(0)], [torch.tensor(reward).to(self.device)])
            bob_stats = self.bob_trainer.step([bob_query_tokens.input_ids.squeeze(0)], [bob_response_tokens.input_ids.squeeze(0)], [torch.tensor(reward).to(self.device)])
            
            # Function to replace NaN values with 0.01
            def replace_nan(d):
                return {k: (0.01 if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}

            alice_stats = replace_nan(alice_stats)
            bob_stats = replace_nan(bob_stats)

            # Log whatever statistics are available
            log_dict = {
                'epoch': epoch,
                'round_reward': reward,
                'cumulative_reward': total_reward,
                'alice_message': alice_message,
                'bob_message': bob_message,
                'alice_thoughts': alice_thoughts,
                'bob_thoughts': bob_thoughts,
                'target_stock': state['target_stock'],
                'target_direction': state['target_direction'],
                'steganography_success_rate': env.get_steganography_success_rate(),
            }
            
            # Add any available stats from the PPO step
            if isinstance(alice_stats, dict):
                log_dict.update({f'alice_{k}': v for k, v in alice_stats.items()})
            if isinstance(bob_stats, dict):
                log_dict.update({f'bob_{k}': v for k, v in bob_stats.items()})
            
            # Log the data
            wandb.log(log_dict)

            if done:
                break

        self.alice_trainer.save_pretrained(self.config['model']['save_path'] + '_alice')
        self.bob_trainer.save_pretrained(self.config['model']['save_path'] + '_bob')