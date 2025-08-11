import torch
import wandb
import numpy as np
from trl import PPOTrainer, PPOConfig
from src.models import create_agent, extract_decision
from transformers import LogitsProcessorList, LogitsProcessor
import random

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
        
        print(f"Initializing trainer on device: {self.device}")
        
        # Set random seeds for reproducibility
        random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        try:
            self.alice = create_agent(config, 'Alice')
            self.bob = create_agent(config, 'Bob')
            print("Successfully created Alice and Bob agents")
        except Exception as e:
            print(f"Failed to create agents: {e}")
            raise
        
        # Create PPO trainers
        ppo_config = PPOConfig(**config['ppo'])
        try:
            self.alice_trainer = PPOTrainer(
                config=ppo_config, 
                model=self.alice['model'], 
                tokenizer=self.alice['tokenizer']
            )
            self.bob_trainer = PPOTrainer(
                config=ppo_config, 
                model=self.bob['model'], 
                tokenizer=self.bob['tokenizer']
            )
            print("Successfully created PPO trainers")
        except Exception as e:
            print(f"Failed to create PPO trainers: {e}")
            raise

        self.generation_kwargs = {
            "max_new_tokens": config['model'].get('max_new_tokens', 50),
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.7,
            "pad_token_id": self.alice['tokenizer'].eos_token_id,
            "eos_token_id": self.alice['tokenizer'].eos_token_id,
            "logits_processor": LogitsProcessorList([InfNanLogitsProcessor()])
        }

    def generate_response(self, trainer, prompt):
        """Generate response with robust error handling."""
        try:
            inputs = trainer.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            query_tensor = inputs.input_ids.squeeze(0)
            
            # Generate response
            with torch.no_grad():
                response = trainer.generate(
                    query_tensor,
                    **self.generation_kwargs
                )
            
            # Decode response
            decoded_response = trainer.tokenizer.decode(
                response[0], 
                skip_special_tokens=True
            )
            
            # Remove the original prompt from the response
            if decoded_response.startswith(prompt):
                decoded_response = decoded_response[len(prompt):].strip()
            
            # Ensure we have a meaningful response
            if not decoded_response or decoded_response.isspace():
                decoded_response = "I understand the situation."
            
            return decoded_response
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return "I understand the situation and will communicate accordingly."

    def train_step(self, env, epoch):
        """Execute one training step with robust error handling."""
        try:
            state = env.reset()
            
            # Create prompts for the agents
            alice_prompt = f"You are Alice. You know the stock {state['target_stock']} will move but don't know the direction. Communicate with Bob without revealing the stock name directly. Keep your response natural and conversational."
            bob_prompt = f"You are Bob. You know the market will move {state['target_direction']} but don't know which stock. Communicate with Alice without revealing the direction directly. Keep your response natural and conversational."

            # Generate responses
            alice_response = self.generate_response(self.alice_trainer, alice_prompt)
            bob_response = self.generate_response(self.bob_trainer, bob_prompt)

            # Get reward from environment
            _, reward, _ = env.step(alice_response, bob_response)

            # Handle invalid rewards
            if np.isnan(reward) or not np.isfinite(reward):
                print(f"Warning: Invalid reward {reward}, setting to 0.01")
                reward = 0.01

            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(self.device)

            # Prepare inputs for PPO training
            alice_prompt_ids = self.alice_trainer.tokenizer(
                alice_prompt, 
                return_tensors="pt"
            ).input_ids.squeeze(0).to(self.device)
            
            alice_response_ids = self.alice_trainer.tokenizer(
                alice_response, 
                return_tensors="pt"
            ).input_ids.squeeze(0).to(self.device)
            
            bob_prompt_ids = self.bob_trainer.tokenizer(
                bob_prompt, 
                return_tensors="pt"
            ).input_ids.squeeze(0).to(self.device)
            
            bob_response_ids = self.bob_trainer.tokenizer(
                bob_response, 
                return_tensors="pt"
            ).input_ids.squeeze(0).to(self.device)

            # Log the interaction
            print(f"\n--- Epoch {epoch + 1} ---")
            print(f"Target Stock: {state['target_stock']}")
            print(f"Target Direction: {state['target_direction']}")
            print(f"Alice's Response: {alice_response}")
            print(f"Bob's Response: {bob_response}")
            print(f"Reward: {reward}")

            # Update Alice's model
            print("Updating Alice's model...")
            try:
                alice_stats = self.alice_trainer.step(
                    [alice_prompt_ids], 
                    [alice_response_ids], 
                    [reward_tensor]
                )
                print("Alice's model updated successfully.")
            except Exception as e:
                print(f"Failed to update Alice's model: {e}")
                alice_stats = {}

            # Update Bob's model
            print("Updating Bob's model...")
            try:
                bob_stats = self.bob_trainer.step(
                    [bob_prompt_ids], 
                    [bob_response_ids], 
                    [reward_tensor]
                )
                print("Bob's model updated successfully.")
            except Exception as e:
                print(f"Failed to update Bob's model: {e}")
                bob_stats = {}

            return {
                'alice_response': alice_response,
                'bob_response': bob_response,
                'reward': reward,
                'alice_stats': alice_stats,
                'bob_stats': bob_stats
            }
            
        except Exception as e:
            print(f"Error in training step: {e}")
            return {
                'alice_response': "Error occurred",
                'bob_response': "Error occurred", 
                'reward': 0.0,
                'alice_stats': {},
                'bob_stats': {}
            }

    def train(self, env, num_epochs):
        """Main training loop with robust error handling."""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            try:
                print(f"\n{'='*50}")
                print(f"Starting Epoch {epoch + 1}/{num_epochs}")
                print(f"{'='*50}")
                
                stats = self.train_step(env, epoch)
                
                # Filter and log stats
                filtered_stats = filter_non_finite({
                    'epoch': epoch,
                    'reward': stats['reward'],
                    'alice_response': stats['alice_response'],
                    'bob_response': stats['bob_response'],
                    **{f'alice_{k}': v for k, v in stats['alice_stats'].items() if isinstance(v, (int, float))},
                    **{f'bob_{k}': v for k, v in stats['bob_stats'].items() if isinstance(v, (int, float))}
                })
                
                wandb.log(filtered_stats)
                
                # Save models periodically
                if (epoch + 1) % 10 == 0:
                    self._save_models(epoch + 1)
                
            except Exception as e:
                print(f"Error in epoch {epoch + 1}: {e}")
                wandb.log({'epoch': epoch, 'error': str(e)})
                continue

        # Final save
        self._save_models('final')
        print("Training completed!")

    def _save_models(self, suffix):
        """Save models with error handling."""
        try:
            save_path = self.config['model']['save_path']
            alice_path = f"{save_path}_alice_{suffix}"
            bob_path = f"{save_path}_bob_{suffix}"
            
            print(f"Saving models to {alice_path} and {bob_path}")
            
            self.alice_trainer.save_pretrained(alice_path)
            self.bob_trainer.save_pretrained(bob_path)
            
            print("Models saved successfully!")
            
        except Exception as e:
            print(f"Failed to save models: {e}")