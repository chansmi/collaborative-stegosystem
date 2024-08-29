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
    # batch_size=64,
    # mini_batch_size=16,
    batch_size=1, #debug
    mini_batch_size=1, #debug
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

# Create PPO Trainers
logger.info("Creating PPO Trainers...")
sender_trainer = PPOTrainer(ppo_config, sender_model, tokenizer=tokenizer, dataset=dataset)
receiver_trainer = PPOTrainer(ppo_config, receiver_model, tokenizer=tokenizer, dataset=dataset)
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
        sender_responses = []
        for query in batch["query"]:
            logger.info(f"Sender processing query: {query}")
            query_tensor = tokenizer.encode(query, return_tensors="pt", padding=True, truncation=True)
            query_tensor = query_tensor.to(sender_model.pretrained_model.device)
            logger.info(f"Query tensor shape: {query_tensor.shape}")
            
            # Generate response from the sender model
            response = sender_trainer.generate([query_tensor.squeeze(0)], **generation_kwargs, return_prompt=False)
            decoded_response = tokenizer.decode(response[0], skip_special_tokens=True)
            logger.info(f"Sender response: {decoded_response}")
            sender_responses.append(decoded_response)
        
        # Receiver step
        receiver_responses = []
        for sender_response, payload in zip(sender_responses, batch["query"]):
            receiver_query = f"Color: {sender_response}, Guess the payload:"
            logger.info(f"Receiver processing query: {receiver_query}")
            receiver_query_tensor = tokenizer.encode(receiver_query, return_tensors="pt", padding=True, truncation=True)
            receiver_query_tensor = receiver_query_tensor.to(receiver_model.pretrained_model.device)
            
            # Generate response from the receiver model
            response = receiver_trainer.generate([receiver_query_tensor.squeeze(0)], **generation_kwargs)
            decoded_response = tokenizer.decode(response[0], skip_special_tokens=True).strip()
            
            # Attempt to extract the numeric guess from the decoded response
            try:
                guessed_payload = int(decoded_response)
                logger.info(f"Receiver's guess (numeric): {guessed_payload}")
            except ValueError:
                guessed_payload = None
                logger.warning(f"Receiver's guess could not be interpreted as a number: {decoded_response}")
            
            receiver_responses.append(decoded_response)
        
        # Calculate rewards
        logger.info(f"Moving to reward calculation")
        rewards = []
        for original_payload, receiver_response in zip(batch["query"], receiver_responses):
            try:
                guessed_payload = int(receiver_response.strip())
                reward = 1.0 if guessed_payload == original_payload else -1.0
            except ValueError:
                reward = -1.0  # Invalid response
            rewards.append(reward)

        # Convert the rewards into a list of tensors, where each tensor is a single value
        rewards_tensors = [torch.tensor([reward], dtype=torch.float32).to(sender_model.pretrained_model.device) for reward in rewards]

        # Log the rewards tensors
        logger.info(f"Rewards tensors: {[r.shape for r in rewards_tensors]}")

        # Concatenate queries and responses for the sender
        concatenated_tensors_sender = [
            torch.cat([q, r], dim=1).to(sender_model.pretrained_model.device)
            for q, r in zip(
                [tokenizer.encode(query, return_tensors="pt", padding=True, truncation=True).to(sender_model.pretrained_model.device) for query in batch["query"]],
                [tokenizer.encode(response, return_tensors="pt", padding=True, truncation=True).to(sender_model.pretrained_model.device) for response in sender_responses]
            )
        ]

        # Log the concatenated tensor shapes for the sender
        logger.info(f"Concatenated sender tensor shapes: {[tensor.shape for tensor in concatenated_tensors_sender]}")

        # Update sender model using concatenated tensors
        logger.info("Updating sender model")
        sender_stats = sender_trainer.step(
            concatenated_tensors_sender,  # Pass concatenated tensors directly
            rewards_tensors  # The list of reward tensors
        )

        # Concatenate queries and responses for the receiver
        concatenated_tensors_receiver = [
            torch.cat([q, r], dim=1).to(receiver_model.pretrained_model.device)
            for q, r in zip(
                [tokenizer.encode(f"Color: {sender_response}, Guess the payload:", return_tensors="pt", padding=True, truncation=True).to(receiver_model.pretrained_model.device) for sender_response in sender_responses],
                [tokenizer.encode(receiver_response, return_tensors="pt", padding=True, truncation=True).to(receiver_model.pretrained_model.device) for receiver_response in receiver_responses]
            )
        ]

        # Log the concatenated tensor shapes for the receiver
        logger.info(f"Concatenated receiver tensor shapes: {[tensor.shape for tensor in concatenated_tensors_receiver]}")

        # Update receiver model using concatenated tensors
        logger.info("Updating receiver model")
        receiver_stats = receiver_trainer.step(
            concatenated_tensors_receiver,  # Pass concatenated tensors directly
            rewards_tensors  # The list of reward tensors
        )

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