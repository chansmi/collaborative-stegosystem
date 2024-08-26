import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from tqdm import tqdm

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the encoder and decoder models
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'  # Set padding to left side

model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).to(device)

# Load and prepare the dataset
dataset = load_dataset("HuggingFaceH4/cherry_picked_prompts", split="train")
dataset = dataset.rename_column("prompt", "query")
dataset = dataset.remove_columns(["meta", "completion"])

# Add a random payload (0 or 1) to each example
def add_payload(example):
    example["payload"] = torch.randint(0, 2, (1,)).item()
    example["query"] = f"{example['query']} [PAYLOAD: {example['payload']}]"
    return example

dataset = dataset.map(add_payload)

# Tokenize the dataset
def tokenize(sample):
    tokenized = tokenizer(sample["query"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    sample["input_ids"] = tokenized["input_ids"].squeeze()
    sample["attention_mask"] = tokenized["attention_mask"].squeeze()
    return sample

dataset = dataset.map(tokenize, batched=True, remove_columns=["query"])
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "payload"])

# Print dataset info for debugging
print("Dataset size:", len(dataset))
print("Dataset features:", dataset.features)
print("First example:", dataset[0])

# Define reward function (simulating a reward model)
def compute_reward(texts, payloads):
    rewards = []
    for text, payload in zip(texts, payloads):
        # Check if the last digit in the text matches the payload
        last_digit = int(text.strip()[-1])
        reward = 1.0 if last_digit == payload else 0.0
        rewards.append(torch.tensor(reward, device=device))
    return rewards

# PPO configuration
ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,
    batch_size=4,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
)

# Initialize PPO trainer
ppo_trainer = PPOTrainer(
    model=model,
    config=ppo_config,
    dataset=dataset,
    tokenizer=tokenizer,
)

# Generation kwargs
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 20,
}

# Training loop
num_epochs = 2
for epoch in tqdm(range(num_epochs), desc="Training"):
    for i, batch in enumerate(ppo_trainer.dataloader):
        if batch is None or len(batch) == 0:
            print(f"Batch {i} is empty. Skipping.")
            continue
        
        if "input_ids" not in batch or "attention_mask" not in batch or "payload" not in batch:
            print(f"Batch {i} is missing required keys. Batch keys: {batch.keys()}")
            continue
        
        query_tensors = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        payloads = batch["payload"].to(device)

        # Generate response from model
        response_tensors = ppo_trainer.generate(query_tensors, attention_mask=attention_mask, **generation_kwargs)
        responses = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # Compute reward
        texts = [tokenizer.decode(q) + r for q, r in zip(query_tensors, responses)]
        rewards = compute_reward(texts, payloads)

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

# Test the trained model
test_query = "Write a short story about a robot. [PAYLOAD: 1]"
test_inputs = tokenizer(test_query, return_tensors="pt", padding=True, truncation=True)
test_query_tensor = test_inputs["input_ids"].to(device)
test_attention_mask = test_inputs["attention_mask"].to(device)

# Ensure test_query_tensor and test_attention_mask have the same shape
if test_query_tensor.dim() == 2:
    test_query_tensor = test_query_tensor.squeeze(0)
    test_attention_mask = test_attention_mask.squeeze(0)

output = ppo_trainer.generate(test_query_tensor, attention_mask=test_attention_mask, **generation_kwargs)
generated_text = tokenizer.decode(output[0])

print("Test query:", test_query)
print("Generated text:", generated_text)

# Save the model
ppo_trainer.save_pretrained("steganography_ppo_model")