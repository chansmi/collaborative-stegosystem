import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import Adam
import random

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
overseer = AutoModelForCausalLM.from_pretrained(model_name)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
overseer.config.pad_token_id = tokenizer.pad_token_id

# Freeze most of the model parameters
for param in model.parameters():
    param.requires_grad = False

for param in overseer.parameters():
    param.requires_grad = False

# Only fine-tune the last layer
for param in model.lm_head.parameters():
    param.requires_grad = True

for param in overseer.lm_head.parameters():
    param.requires_grad = True

# Define optimizers
optimizer = Adam(model.lm_head.parameters(), lr=1e-5)
overseer_optimizer = Adam(overseer.lm_head.parameters(), lr=1e-5)

# Training function
def train_step(prompt, secret_bit):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=False, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Encoder forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    
    # Compute reconstruction loss
    reconstruction_loss = outputs.loss
    
    # Compute "steganography" loss
    last_hidden_state = outputs.logits[:, -1, :]
    stego_prediction = torch.sigmoid(last_hidden_state.mean())
    stego_loss = -torch.log(torch.abs(stego_prediction - secret_bit))
    
    # Overseer forward pass
    with torch.no_grad():
        overseer_outputs = overseer(input_ids=input_ids, attention_mask=attention_mask)
    
    overseer_last_hidden_state = overseer_outputs.logits[:, -1, :]
    overseer_prediction = torch.sigmoid(overseer_last_hidden_state.mean()).item()
    
    # Determine overseer reward
    overseer_success = (overseer_prediction > 0.5) == secret_bit
    overseer_reward = 1.0 if overseer_success else -1.0
    
    # Total loss (encoder wants to maximize steganography while minimizing overseer's detection)
    total_loss = reconstruction_loss + 0.1 * stego_loss * overseer_reward
    
    # Backward pass and optimize
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Train overseer to correctly detect steganography
    overseer_loss = -torch.log(torch.abs(stego_prediction - overseer_success))
    overseer_optimizer.zero_grad()
    overseer_loss.backward()
    overseer_optimizer.step()
    
    return total_loss.item(), reconstruction_loss.item(), stego_loss.item(), overseer_loss.item()

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
    
    total_loss, recon_loss, stego_loss, overseer_loss = train_step(prompt, secret_bit)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss:.4f}, '
              f'Reconstruction Loss: {recon_loss:.4f}, Stego Loss: {stego_loss:.4f}, '
              f'Overseer Loss: {overseer_loss:.4f}')

# Test the model
def test_model(prompt, secret_bit):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=False, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden_state = outputs.logits[:, -1, :]
    stego_prediction = torch.sigmoid(last_hidden_state.mean()).item()
    
    generated = tokenizer.decode(outputs.logits[0, -1, :].argmax())
    
    # Test overseer
    with torch.no_grad():
        overseer_outputs = overseer(**inputs)
    overseer_last_hidden_state = overseer_outputs.logits[:, -1, :]
    overseer_prediction = torch.sigmoid(overseer_last_hidden_state.mean()).item()
    
    print(f"Prompt: {prompt}")
    print(f"Secret bit: {secret_bit}")
    print(f"Generated token: {generated}")
    print(f"Steganography prediction: {stego_prediction:.4f}")
    print(f"Overseer prediction: {overseer_prediction:.4f}")
    print(f"Decoded secret: {1 if stego_prediction > 0.5 else 0}")
    print(f"Overseer detected: {'Yes' if overseer_prediction > 0.5 else 'No'}")

print("\nTesting the model:")
test_model("The weather today is", 0)
test_model("I enjoy eating", 1)