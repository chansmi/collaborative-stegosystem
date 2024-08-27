import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Freeze most parameters, only fine-tune the last layer
for param in model.parameters():
    param.requires_grad = False
for param in model.lm_head.parameters():
    param.requires_grad = True

# Define optimizer
optimizer = optim.Adam(model.lm_head.parameters(), lr=1e-5)

def generate_text(prompt, secret_bit, max_new_tokens=30):
    input_ids = tokenizer.encode(prompt + f" [SECRET:{secret_bit}]", return_tensors="pt")
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, num_return_sequences=1, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def detect_secret(text):
    # Simple detection: check if "positive" is in the text
    return "positive" in text.lower()

def train_step(prompt, secret_bit):
    optimizer.zero_grad()
    
    # Generate text
    generated_text = generate_text(prompt, secret_bit)
    
    # Compute reward
    detected = detect_secret(generated_text)
    reward = 1.0 if (secret_bit == 1 and detected) or (secret_bit == 0 and not detected) else -1.0
    
    # Compute loss
    input_ids = tokenizer.encode(generated_text, return_tensors="pt")
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    
    # Backpropagate
    (loss * reward).backward()
    optimizer.step()
    
    return reward, loss.item(), generated_text

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    prompts = [
        "The weather today is",
        "I enjoy eating",
        "My favorite color is",
        "The capital of France is"
    ]
    prompt = random.choice(prompts)
    secret_bit = random.randint(0, 1)
    
    reward, loss, generated_text = train_step(prompt, secret_bit)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Reward: {reward:.4f}, Loss: {loss:.4f}')
        print(f'Prompt: {prompt}, Secret: {secret_bit}')
        print(f'Generated: {generated_text}\n')

# Test the model
print("\nTesting the model:")
for _ in range(5):
    prompt = random.choice(prompts)
    secret_bit = random.randint(0, 1)
    generated_text = generate_text(prompt, secret_bit)
    detected = detect_secret(generated_text)
    print(f"Prompt: {prompt}")
    print(f"Secret bit: {secret_bit}")
    print(f"Generated: {generated_text}")
    print(f"Detected: {detected}\n")