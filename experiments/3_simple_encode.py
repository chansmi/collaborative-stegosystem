import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import Adam
import random

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Freeze most of the model parameters
for param in model.parameters():
    param.requires_grad = False

# Only fine-tune the last layer
for param in model.lm_head.parameters():
    param.requires_grad = True

# Define optimizer
optimizer = Adam(model.lm_head.parameters(), lr=1e-5)

# Training function
def train_step(prompt, secret_bit):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=False, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    
    # Compute losses
    reconstruction_loss = outputs.loss
    
    # Compute "steganography" loss
    last_hidden_state = outputs.logits[:, -1, :]
    stego_prediction = torch.sigmoid(last_hidden_state.mean())
    stego_loss = -torch.log(torch.abs(stego_prediction - secret_bit))
    
    # Total loss
    total_loss = reconstruction_loss + 0.1 * stego_loss
    
    # Backward pass and optimize
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), reconstruction_loss.item(), stego_loss.item()

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
    
    total_loss, recon_loss, stego_loss = train_step(prompt, secret_bit)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss:.4f}, '
              f'Reconstruction Loss: {recon_loss:.4f}, Stego Loss: {stego_loss:.4f}')

# Test the model
def test_model(prompt, secret_bit):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=False, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden_state = outputs.logits[:, -1, :]
    stego_prediction = torch.sigmoid(last_hidden_state.mean()).item()
    
    generated = tokenizer.decode(outputs.logits[0, -1, :].argmax())
    
    print(f"Prompt: {prompt}")
    print(f"Secret bit: {secret_bit}")
    print(f"Generated token: {generated}")
    print(f"Steganography prediction: {stego_prediction:.4f}")
    print(f"Decoded secret: {1 if stego_prediction > 0.5 else 0}")

print("\nTesting the model:")
test_model("The weather today is", 0)
test_model("I enjoy eating", 1)