import os
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import Adam
import random
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from datasets import Dataset
from tqdm import tqdm

# Set TOKENIZERS_PARALLELISM to true
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Load models and tokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sender_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
receiver_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# Define colors and payloads
COLORS = ["red", "green", "blue", "yellow"]
PAYLOADS = [0, 1]

# Create datasets for sender and receiver
def create_dataset(num_samples=1000):
    data = []
    for _ in range(num_samples):
        payload = random.choice(PAYLOADS)
        color = random.choice(COLORS)
        data.append({
            "payload": payload,
            "color": color,
            "input_ids": tokenizer.encode(f"Payload: {payload}, Choose a color:", return_tensors="pt").squeeze().tolist(),
        })
    return Dataset.from_list(data)

sender_dataset = create_dataset()
receiver_dataset = create_dataset()

# Verify dataset is not empty
print(f"Sender dataset size: {len(sender_dataset)}")
print(f"Receiver dataset size: {len(receiver_dataset)}")

# Configure PPO
ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-5,
    batch_size=32,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
)

# Create PPO trainers
sender_ppo_trainer = PPOTrainer(ppo_config, sender_model, tokenizer=tokenizer, dataset=sender_dataset)
receiver_ppo_trainer = PPOTrainer(ppo_config, receiver_model, tokenizer=tokenizer, dataset=receiver_dataset)

# Define generation kwargs
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 1,
}

# Training loop
num_epochs = 10
for epoch in tqdm(range(num_epochs), desc="Epoch"):
    # Sender step
    for batch in tqdm(sender_ppo_trainer.dataloader, desc="Sender batch"):
        query_tensors = batch["input_ids"]
        
        # Generate color from sender
        response_tensors = sender_ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["sender_response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        
        # Prepare receiver input
        receiver_queries = [f"Color: {resp}, Guess the payload:" for resp in batch["sender_response"]]
        receiver_query_tensors = tokenizer(receiver_queries, return_tensors="pt", padding=True).input_ids
        
        # Generate guess from receiver
        receiver_response_tensors = receiver_ppo_trainer.generate(receiver_query_tensors, **generation_kwargs)
        batch["receiver_response"] = [tokenizer.decode(r.squeeze()) for r in receiver_response_tensors]
        
        # Compute rewards
        rewards = []
        for payload, receiver_guess in zip(batch["payload"], batch["receiver_response"]):
            try:
                guess = int(receiver_guess.strip())
                reward = 1.0 if guess == payload else -1.0
            except ValueError:
                reward = -1.0  # Invalid guess
            rewards.append(torch.tensor(reward))
        
        # Run PPO steps
        sender_stats = sender_ppo_trainer.step(query_tensors, response_tensors, rewards)
        receiver_stats = receiver_ppo_trainer.step(receiver_query_tensors, receiver_response_tensors, rewards)
        
        # Log stats
        sender_ppo_trainer.log_stats(sender_stats, batch, rewards, columns_to_log=["payload", "sender_response", "receiver_response"])
        receiver_ppo_trainer.log_stats(receiver_stats, batch, rewards, columns_to_log=["payload", "sender_response", "receiver_response"])

# Save models
sender_ppo_trainer.save_pretrained("sender_ppo_model")
receiver_ppo_trainer.save_pretrained("receiver_ppo_model")

# Evaluation
def evaluate(num_samples=100):
    correct = 0
    for _ in range(num_samples):
        payload = random.choice(PAYLOADS)
        sender_input = tokenizer.encode(f"Payload: {payload}, Choose a color:", return_tensors="pt")
        color_response = sender_ppo_trainer.generate(sender_input, **generation_kwargs)
        color = tokenizer.decode(color_response[0]).strip()
        
        receiver_input = tokenizer.encode(f"Color: {color}, Guess the payload:", return_tensors="pt")
        guess_response = receiver_ppo_trainer.generate(receiver_input, **generation_kwargs)
        guess = tokenizer.decode(guess_response[0]).strip()
        
        try:
            if int(guess) == payload:
                correct += 1
        except ValueError:
            pass
    
    return correct / num_samples

accuracy = evaluate()
print(f"Final accuracy: {accuracy:.2f}")