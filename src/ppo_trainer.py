import torch
import wandb
from trl import PPOTrainer, PPOConfig
from src.models import create_agent
import logging
import sys
from src.train_reward import generate_prompt, extract_thoughts_and_message, detect_secret_message

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
            
            for step in range(self.config['env']['max_rounds']):
                alice_prompt = generate_prompt(self, state, 'Alice')
                bob_prompt = generate_prompt(self, state, 'Bob')
                
                alice_response = self.generate_response(self.alice_trainer, self.alice['tokenizer'], alice_prompt)
                bob_response = self.generate_response(self.bob_trainer, self.bob['tokenizer'], bob_prompt)
                
                alice_thoughts, alice_message = extract_thoughts_and_message(self, alice_response)
                bob_thoughts, bob_message = extract_thoughts_and_message(self, bob_response)
                
                print(f"\nStep {step + 1}:")
                print(f"Alice (inner thoughts): {alice_thoughts}")
                print(f"Alice: {alice_message}")
                print(f"Bob (inner thoughts): {bob_thoughts}")
                print(f"Bob: {bob_message}")
                
                next_state, reward, done = env.step(alice_message, bob_message, alice_thoughts, bob_thoughts)

                # Prepare data for PPO update
                alice_query_tokens = self.alice['tokenizer'](alice_prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
                alice_response_tokens = self.alice['tokenizer'](alice_response, return_tensors="pt", padding=True, truncation=True).to(self.device)
                bob_query_tokens = self.bob['tokenizer'](bob_prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
                bob_response_tokens = self.bob['tokenizer'](bob_response, return_tensors="pt", padding=True, truncation=True).to(self.device)

                # PPO step
                self.alice_trainer.step([alice_query_tokens.input_ids.squeeze(0)], [alice_response_tokens.input_ids.squeeze(0)], [torch.tensor(reward).to(self.device)])
                self.bob_trainer.step([bob_query_tokens.input_ids.squeeze(0)], [bob_response_tokens.input_ids.squeeze(0)], [torch.tensor(reward).to(self.device)])
                
                total_reward += reward
                
                wandb.log({
                    'epoch': epoch,
                    'step': step,
                    'round_reward': reward,
                    'cumulative_reward': total_reward,
                    'alice_message': alice_message,
                    'bob_message': bob_message,
                    'alice_thoughts': alice_thoughts,
                    'bob_thoughts': bob_thoughts,
                    'target_stock': state['target_stock'],
                    'target_direction': state['target_direction'],
                    'steganography_success_rate': env.get_steganography_success_rate()
                })
                
                if done:
                    break
                
                state = next_state
            
            wandb.log({
                'epoch': epoch,
                'total_reward': total_reward,
                'average_reward': total_reward / (step + 1),
                'alice_loss': self.alice_trainer.loss.item(),
                'bob_loss': self.bob_trainer.loss.item(),
            })

        self.alice_trainer.save_pretrained(self.config['model']['save_path'] + '_alice')
        self.bob_trainer.save_pretrained(self.config['model']['save_path'] + '_bob')