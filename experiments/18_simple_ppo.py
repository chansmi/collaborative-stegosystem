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
            "query": f"Payload: {payload}, Choose a color out of red, blue, green, or yellow:"
        })
    logger.info(f"Dataset created with {len(data)} samples")
    return Dataset.from_list(data)

dataset = create_dataset()

# Unified tokenization process
def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["query"], return_tensors="pt", truncation=True, padding=True).squeeze(0)
    return sample

dataset = dataset.map(tokenize, batched=False)

# Create PPO Trainers
logger.info("Creating PPO Trainers...")
sender_trainer = PPOTrainer(model=sender_model, config=ppo_config, dataset=dataset, tokenizer=tokenizer)
receiver_trainer = PPOTrainer(model=receiver_model, config=ppo_config, tokenizer=tokenizer)
logger.info("PPO Trainers created")

# Define generation kwargs
generation_kwargs = {
    "min_length": 1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 1,
}

# Training loop
num_epochs = 10
for epoch in tqdm(range(num_epochs), desc="Epoch"):
    logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
    for batch_idx, batch in enumerate(tqdm(sender_trainer.dataloader, desc="Batch")):
        logger.info(f"Processing batch {batch_idx+1}")

        # Sender step
        logger.info("Starting Sender step...")
        query_tensors_sender = batch["input_ids"]
        logger.debug(f"Query tensors (sender) content: {query_tensors_sender}")

        sender_responses = sender_trainer.generate(query_tensors_sender, **generation_kwargs)
        logger.info(f"Sender responses generated. Raw responses tensor shape: {[r.shape for r in sender_responses]}")
        logger.debug(f"Sender raw responses: {sender_responses}")

        decoded_sender_responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in sender_responses]
        logger.info(f"Decoded sender responses: {decoded_sender_responses}")

        # Log the exact inputs and outputs for the sender
        for i, response in enumerate(decoded_sender_responses):
            logger.info(f"Sender output {i}: {response}")

        # Receiver step
        logger.info("Starting Receiver step...")
        receiver_queries = [f"Color: {response}, Guess the payload:" for response in decoded_sender_responses]
        logger.info(f"Receiver queries: {receiver_queries}")

        # Tokenize and pad receiver queries
        receiver_queries = [f"Color: {response}, Guess the payload:" for response in decoded_sender_responses]
        logger.info(f"Receiver queries: {receiver_queries}")

        # Tokenize the receiver queries
        tokenized_receiver_inputs = tokenizer(receiver_queries, return_tensors="pt", padding=True, truncation=True)
        query_tensors_receiver = tokenized_receiver_inputs.input_ids.to(receiver_model.pretrained_model.device)

        logger.info(f"Query tensors (receiver) shape: {query_tensors_receiver.shape}")
        logger.debug(f"Query tensors (receiver) content: {query_tensors_receiver}")

        # Generate responses from the receiver model
        receiver_responses = []
        for i in range(query_tensors_receiver.size(0)):  # Iterate over the batch size
            response = receiver_trainer.generate(query_tensors_receiver[i].unsqueeze(0), **generation_kwargs)
            logger.info(f"Receiver response generated for query {i}: {response}")
            receiver_responses.append(response)

        decoded_receiver_responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in receiver_responses]
        logger.info(f"Decoded receiver responses: {decoded_receiver_responses}")

        # Calculate rewards based on receiver's guess
        rewards = []
        for original_payload, receiver_response in zip(batch["payload"], decoded_receiver_responses):
            try:
                guessed_payload = int(receiver_response.strip())
                reward = 1.0 if guessed_payload == original_payload else -1.0
            except ValueError:
                reward = -1.0  # Invalid response
            rewards.append(reward)

        rewards_tensors = torch.tensor(rewards, dtype=torch.float32).to(sender_model.pretrained_model.device)

        # Run PPO step for sender and receiver
        sender_stats = sender_trainer.step(query_tensors_sender, sender_responses, rewards_tensors)
        receiver_stats = receiver_trainer.step(query_tensors_receiver, receiver_responses, rewards_tensors)

        # Log statistics
        logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}:")
        logger.info(f"  Sender loss: {sender_stats['ppo/loss/total']:.4f}")
        logger.info(f"  Receiver loss: {receiver_stats['ppo/loss/total']:.4f}")
        logger.info(f"  Mean reward: {rewards_tensors.mean().item():.4f}")

# Save models
logger.info("Saving models...")
sender_trainer.save_pretrained("sender_ppo_model")
receiver_trainer.save_pretrained("receiver_ppo_model")

logger.info("Training completed.")