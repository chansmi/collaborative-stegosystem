import torch
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, set_seed
from tqdm import tqdm

# Step 1: Define PPO Config and Override model to GPT-2
ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1e-5,
    batch_size=2,
    mini_batch_size=2,
    ppo_epochs=1
)

# Set seed for reproducibility
set_seed(42)

# Load GPT-2 model with value head (for PPO training)
model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name)
tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)

# Ensure tokenizer has padding token
tokenizer.pad_token = tokenizer.eos_token

# Step 2: Load a small dataset for both train and validation
train_dataset = load_dataset("imdb", split="train[:100]")
valid_dataset = load_dataset("imdb", split="test[:50]")

# Preprocess data (tokenizing and truncating)
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)
valid_dataset = valid_dataset.map(preprocess_function, batched=True)

# Set format for PyTorch tensors
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Step 3: Initialize PPOTrainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
    dataset=train_dataset
)

# Step 4: PPO Training loop with sentiment analysis for rewards
sentiment_pipe = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)

# Arguments for generation
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}

# Run training with PPO steps
for batch in tqdm(ppo_trainer.dataloader):
    query_tensors = batch["input_ids"]

    # Generate responses using GPT-2 model
    response_tensors, ref_response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
    )
    batch["response"] = tokenizer.batch_decode(response_tensors)
    batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

    # Compute sentiment score (this will serve as the reward for PPO)
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, return_all_scores=True)
    rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])

# Step 5: Validation process
# After PPO training, let's evaluate on the validation set
print("\nRunning validation...")

validation_generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}

# Evaluate the model on the validation set and print the results
for batch in tqdm(valid_dataset):
    input_ids = batch["input_ids"].unsqueeze(0)  # Single input, so we add batch dimension
    generated_output = model.generate(input_ids, **validation_generation_kwargs)
    decoded_output = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    print(f"Input: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
    print(f"Generated Output: {decoded_output}")