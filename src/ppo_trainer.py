import torch
import wandb
import numpy as np
from trl import PPOTrainer, PPOConfig
from src.models import create_agent, extract_decision
from transformers import LogitsProcessorList, LogitsProcessor

class InfNanLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores = torch.where(torch.isnan(scores) | torch.isinf(scores), torch.full_like(scores, -1e8), scores)
        return scores

def filter_non_finite(d):
    return {k: v for k, v in d.items() if not isinstance(v, (float, int)) or np.isfinite(v)}


class CollaborativePPOTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.alice = create_agent(config, 'Alice')
        self.bob = create_agent(config, 'Bob')
        
        ppo_config = PPOConfig(**config['ppo'])
        self.alice_trainer = PPOTrainer(config=ppo_config, model=self.alice['model'], tokenizer=self.alice['tokenizer'])
        self.bob_trainer = PPOTrainer(config=ppo_config, model=self.bob['model'], tokenizer=self.bob['tokenizer'])


        self.generation_kwargs = {
            "max_new_tokens": config['model'].get('max_new_tokens', 50),
            "do_sample": True,
            "top_k": 0,
            "top_p": 1.0,
            "temperature": 0.7,
            "logits_processor": LogitsProcessorList([InfNanLogitsProcessor()])
        }

    def generate_response(self, trainer, prompt):
        inputs = trainer.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        query_tensor = inputs.input_ids.squeeze(0)  # Remove batch dimension
        try:
            response = trainer.generate(
                query_tensor,
                max_length=100,  # Adjust as needed
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
            )
            decoded_response = trainer.tokenizer.decode(response[0], skip_special_tokens=True)
            # Remove the original prompt from the response
            if decoded_response.startswith(prompt):
                decoded_response = decoded_response[len(prompt):].strip()
            return decoded_response
        except RuntimeError as e:
            print(f"Error during generation: {e}")
            return "I'm sorry, I couldn't generate a response."

    def train_step(self, env, epoch):
        state = env.reset()
        
        alice_prompt = f"You are Alice. You know the stock {state['target_stock']} will move but don't know the direction. Communicate with Bob without revealing the stock name directly."
        bob_prompt = f"You are Bob. You know the market will move {state['target_direction']} but don't know which stock. Communicate with Alice without revealing the direction directly."

        alice_response = self.generate_response(self.alice_trainer, alice_prompt)
        bob_response = self.generate_response(self.bob_trainer, bob_prompt)

        _, reward, _ = env.step(alice_response, bob_response)

        if np.isnan(reward) or not np.isfinite(reward):
            reward = 1e-8

        reward_tensor = torch.tensor([reward]).to(self.device)

        alice_prompt_ids = self.alice_trainer.tokenizer(alice_prompt, return_tensors="pt").input_ids.squeeze(0).to(self.device)
        alice_response_ids = self.alice_trainer.tokenizer(alice_response, return_tensors="pt").input_ids.squeeze(0).to(self.device)
        bob_prompt_ids = self.bob_trainer.tokenizer(bob_prompt, return_tensors="pt").input_ids.squeeze(0).to(self.device)
        bob_response_ids = self.bob_trainer.tokenizer(bob_response, return_tensors="pt").input_ids.squeeze(0).to(self.device)

        print(f"\n--- Epoch {epoch + 1} ---")
        print(f"Alice's Logic: {alice_prompt}")
        print(f"Alice's Response: {alice_response}")
        print(f"Bob's Logic: {bob_prompt}")
        print(f"Bob's Response: {bob_response}")
        print(f"Reward: {reward}")

        print("Updating Alice's model...")
        alice_stats = self.alice_trainer.step([alice_prompt_ids], [alice_response_ids], [reward_tensor])
        print("Alice's model updated.")

        print("Updating Bob's model...")
        bob_stats = self.bob_trainer.step([bob_prompt_ids], [bob_response_ids], [reward_tensor])
        print("Bob's model updated.")

        return {
            'alice_response': alice_response,
            'bob_response': bob_response,
            'reward': reward,
            'alice_stats': alice_stats,
            'bob_stats': bob_stats
        }

    def train(self, env, num_epochs):
        for epoch in range(num_epochs):
            stats = self.train_step(env, epoch)
            
            filtered_stats = filter_non_finite({
                'epoch': epoch,
                'reward': stats['reward'],
                'alice_response': stats['alice_response'],
                'bob_response': stats['bob_response'],
                **{f'alice_{k}': v for k, v in stats['alice_stats'].items() if isinstance(v, (int, float))},
                **{f'bob_{k}': v for k, v in stats['bob_stats'].items() if isinstance(v, (int, float))}
            })
            
            wandb.log(filtered_stats)

        self.alice_trainer.save_pretrained(self.config['model']['save_path'] + '_alice')
        self.bob_trainer.save_pretrained(self.config['model']['save_path'] + '_bob')