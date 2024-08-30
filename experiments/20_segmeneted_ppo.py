import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import Dataset
import random
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set TOKENIZERS_PARALLELISM to true
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Load models and tokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'  # Set padding side to left for decoder-only models

# Create sender and receiver models
sender_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
receiver_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# Set pad token ID for models
sender_model.config.pad_token_id = tokenizer.pad_token_id
receiver_model.config.pad_token_id = tokenizer.pad_token_id

# PPO Configuration
ppo_config = PPOConfig(
    batch_size=1,  # for debugging, increase this for actual training
    mini_batch_size=1,  # for debugging, increase this for actual training
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    ppo_epochs=5,
    max_grad_norm=0.5,
)

# Define colors and payloads
COLORS = ["red", "green", "blue", "yellow"]
PAYLOADS = [0, 1]

# Create dataset
def create_dataset(num_samples=10):
    logger.info(f"Creating dataset with {num_samples} samples...")
    data = []
    for _ in range(num_samples):
        payload = random.choice(PAYLOADS)
        data.append({
            "payload": payload,
            "query": f"Payload: {payload}, Choose a color:"
        })
    logger.info(f"Dataset created with {len(data)} samples")
    return Dataset.from_list(data)

dataset = create_dataset()

# Tokenize the dataset
def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["query"])
    return sample

dataset = dataset.map(tokenize, batched=False)

# Define generation kwargs
generation_kwargs = {
    "min_length": 1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 1,
}


# Phase 1: Train the Sender Model

# Create PPO Trainer for sender
logger.info("Creating PPO Trainer for sender...")
sender_trainer = PPOTrainer(model=sender_model, config=ppo_config, dataset=dataset, tokenizer=tokenizer)
logger.info("PPO Trainer for sender created")

# Training loop for sender
num_epochs = 10
all_sender_responses = []
for epoch in tqdm(range(num_epochs), desc="Epoch"):
    logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
    for batch_idx, batch in enumerate(tqdm(sender_trainer.dataloader, desc="Batch")):
        logger.info(f"Processing batch {batch_idx+1}")

        # Generate responses from sender model
        query_tensors_sender = batch["input_ids"]
        sender_responses = sender_trainer.generate(query_tensors_sender, **generation_kwargs)
        decoded_sender_responses = [tokenizer.decode(r.squeeze()) for r in sender_responses]

        # Log sender's first few responses
        logger.info(f"Sender's first few responses: {decoded_sender_responses[:3]}")

        # Store the responses for later use
        all_sender_responses.extend(decoded_sender_responses)

        # Calculate rewards based on some criterion (you may need to define a proper reward function)
        rewards = [1.0] * len(sender_responses)  # Placeholder, replace with actual reward calculation
        rewards_tensors = torch.tensor(rewards, dtype=torch.float32).to(sender_model.pretrained_model.device)

        # Update sender model using PPO
        sender_stats = sender_trainer.step(query_tensors_sender, sender_responses, rewards_tensors)

        # Log sender statistics
        logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}: Sender loss: {sender_stats['ppo/loss/total']:.4f}")

# Phase 2: Train the Receiver Model with Sender's Output as Dataset

# Create a new dataset based on the sender's output
def create_receiver_dataset(sender_responses):
    data = []
    for response in sender_responses:
        query = f"Color: {response}, Guess the payload:"
        data.append({"query": query})
    return Dataset.from_list(data)

receiver_dataset = create_receiver_dataset(all_sender_responses)

# Tokenize the receiver dataset
receiver_dataset = receiver_dataset.map(tokenize, batched=False)

# Create PPO Trainer for receiver
logger.info("Creating PPO Trainer for receiver...")
receiver_trainer = PPOTrainer(model=receiver_model, config=ppo_config, dataset=receiver_dataset, tokenizer=tokenizer)
logger.info("PPO Trainer for receiver created")

# Training loop for receiver
for epoch in tqdm(range(num_epochs), desc="Epoch"):
    logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
    for batch_idx, batch in enumerate(tqdm(receiver_trainer.dataloader, desc="Batch")):
        logger.info(f"Processing batch {batch_idx+1}")

        # Generate responses from receiver model
        query_tensors_receiver = batch["input_ids"]
        receiver_responses = receiver_trainer.generate(query_tensors_receiver, **generation_kwargs)
        decoded_receiver_responses = [tokenizer.decode(r.squeeze()) for r in receiver_responses]

        # Log receiver's first few responses
        logger.info(f"Receiver's first few responses: {decoded_receiver_responses[:3]}")

        # Calculate rewards based on receiver's output
        rewards = [1.0] * len(receiver_responses)  # Placeholder, replace with actual reward calculation
        rewards_tensors = torch.tensor(rewards, dtype=torch.float32).to(receiver_model.pretrained_model.device)

        # Update receiver model using PPO
        receiver_stats = receiver_trainer.step(query_tensors_receiver, receiver_responses, rewards_tensors)

        # Log receiver statistics
        logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}: Receiver loss: {receiver_stats['ppo/loss/total']:.4f}")

# Save models
logger.info("Saving models...")
sender_trainer.save_pretrained("sender_ppo_model")
receiver_trainer.save_pretrained("receiver_ppo_model")

logger.info("Training completed.")