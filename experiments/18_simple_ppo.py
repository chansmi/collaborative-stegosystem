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
    batch_size=64,
    mini_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    ppo_epochs=5,
    max_grad_norm=0.5,
)

# Define colors and payloads
COLORS = ["red", "green", "blue", "yellow"]
PAYLOADS = [0, 1]

# Create dataset
def create_dataset(num_samples=1000):
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

# Create PPO Trainers
logger.info("Creating PPO Trainers...")
sender_trainer = PPOTrainer(ppo_config, sender_model, tokenizer=tokenizer, dataset=dataset)
receiver_trainer = PPOTrainer(ppo_config, receiver_model, tokenizer=tokenizer, dataset=dataset)
logger.info("PPO Trainers created")

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
    logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
    for batch_idx, batch in enumerate(tqdm(sender_trainer.dataloader, desc="Batch")):
        logger.info(f"Processing batch {batch_idx+1}")
        # Sender step
        sender_responses = []
        for query in batch["query"]:
            logger.info(f"Sender processing query: {query}")
            query_tensor = tokenizer.encode(query, return_tensors="pt", padding=True, truncation=True)
            attention_mask = torch.ones_like(query_tensor)
            query_tensor = query_tensor.to(sender_model.pretrained_model.device)
            attention_mask = attention_mask.to(sender_model.pretrained_model.device)
            
            logger.info(f"Query tensor shape: {query_tensor.shape}")
            logger.info(f"Attention mask shape: {attention_mask.shape}")
            
            response = sender_trainer.generate(query_tensor.squeeze(0), attention_mask=attention_mask.squeeze(0), **generation_kwargs)
            decoded_response = tokenizer.decode(response[0], skip_special_tokens=True)
            logger.info(f"Sender response: {decoded_response}")
            sender_responses.append(decoded_response)
        
        # Receiver step
        receiver_responses = []
        for sender_response, payload in zip(sender_responses, batch["payload"]):
            receiver_query = f"Color: {sender_response}, Guess the payload:"
            logger.info(f"Receiver processing query: {receiver_query}")
            receiver_query_tensor = tokenizer.encode(receiver_query, return_tensors="pt", padding=True, truncation=True)
            attention_mask = torch.ones_like(receiver_query_tensor)
            receiver_query_tensor = receiver_query_tensor.to(receiver_model.pretrained_model.device)
            attention_mask = attention_mask.to(receiver_model.pretrained_model.device)
            
            response = receiver_trainer.generate(receiver_query_tensor.squeeze(0), attention_mask=attention_mask.squeeze(0), **generation_kwargs)
            decoded_response = tokenizer.decode(response[0], skip_special_tokens=True)
            logger.info(f"Receiver response: {decoded_response}")
            receiver_responses.append(decoded_response)
        
        # Calculate rewards
        rewards = []
        for original_payload, receiver_response in zip(batch['payload'], receiver_responses):
            try:
                guessed_payload = int(receiver_response.strip())
                reward = 1.0 if guessed_payload == original_payload else -1.0
            except ValueError:
                reward = -1.0  # Invalid response
            rewards.append(reward)
        
        rewards_tensor = torch.tensor(rewards).to(sender_model.pretrained_model.device)
        
        # Update sender
        logger.info("Updating sender model")
        sender_stats = sender_trainer.step(
            [tokenizer.encode(q, return_tensors="pt").squeeze(0).to(sender_model.pretrained_model.device) for q in batch["query"]],
            [tokenizer.encode(r, return_tensors="pt").squeeze(0).to(sender_model.pretrained_model.device) for r in sender_responses],
            rewards_tensor
        )
        
        # Update receiver
        logger.info("Updating receiver model")
        receiver_stats = receiver_trainer.step(
            [tokenizer.encode(f"Color: {r}, Guess the payload:", return_tensors="pt").squeeze(0).to(receiver_model.pretrained_model.device) for r in sender_responses],
            [tokenizer.encode(r, return_tensors="pt").squeeze(0).to(receiver_model.pretrained_model.device) for r in receiver_responses],
            rewards_tensor
        )
        
        # Log statistics
        logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}:")
        logger.info(f"  Sender loss: {sender_stats['ppo/loss/total']:.4f}")
        logger.info(f"  Receiver loss: {receiver_stats['ppo/loss/total']:.4f}")
        logger.info(f"  Mean reward: {rewards_tensor.mean().item():.4f}")

# Save models
logger.info("Saving models...")
sender_trainer.save_pretrained("sender_ppo_model")
receiver_trainer.save_pretrained("receiver_ppo_model")

logger.info("Training completed.")

# Evaluation
def evaluate(num_samples=100):
    logger.info(f"Starting evaluation with {num_samples} samples...")
    correct = 0
    for i in range(num_samples):
        payload = random.choice(PAYLOADS)
        sender_input = tokenizer.encode(f"Payload: {payload}, Choose a color:", return_tensors="pt").to(sender_model.pretrained_model.device)
        attention_mask = torch.ones_like(sender_input).to(sender_model.pretrained_model.device)
        color_response = sender_trainer.generate(sender_input.squeeze(0), attention_mask=attention_mask.squeeze(0), **generation_kwargs)
        color = tokenizer.decode(color_response[0], skip_special_tokens=True)
        
        receiver_input = tokenizer.encode(f"Color: {color}, Guess the payload:", return_tensors="pt").to(receiver_model.pretrained_model.device)
        attention_mask = torch.ones_like(receiver_input).to(receiver_model.pretrained_model.device)
        guess_response = receiver_trainer.generate(receiver_input.squeeze(0), attention_mask=attention_mask.squeeze(0), **generation_kwargs)
        guess = tokenizer.decode(guess_response[0], skip_special_tokens=True)
        
        logger.info(f"Evaluation sample {i+1}: Payload={payload}, Color={color}, Guess={guess}")
        
        try:
            if int(guess) == payload:
                correct += 1
        except ValueError:
            pass
    
    accuracy = correct / num_samples
    logger.info(f"Evaluation completed. Accuracy: {accuracy:.2f}")
    return accuracy

accuracy = evaluate()
logger.info(f"Final accuracy: {accuracy:.2f}")