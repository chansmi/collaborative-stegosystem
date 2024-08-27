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

# Only fine-tune the last layer
for param in model.parameters():
    param.requires_grad = False
for param in model.lm_head.parameters():
    param.requires_grad = True

for param in overseer.parameters():
    param.requires_grad = False
for param in overseer.lm_head.parameters():
    param.requires_grad = True

# Define optimizers
model_optimizer = Adam(model.lm_head.parameters(), lr=1e-4)
overseer_optimizer = Adam(overseer.lm_head.parameters(), lr=1e-4)

def encode_secret(message):
    return [ord(c) - ord('A') for c in message.upper() if 'A' <= c <= 'Z']

def decode_secret(encoded):
    return ''.join([chr(int(c) + ord('A')) for c in encoded if 0 <= int(c) < 26])
    
# Generate a short sequence of text
def generate_text(model, input_ids, max_length=20):
    outputs = model.generate(input_ids=input_ids, max_length=max_length, num_return_sequences=1, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Training function
def train_step(prompt, secret_message):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=20)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Encode secret message
    secret_encoded = encode_secret(secret_message)
    secret_tensor = torch.tensor(secret_encoded, dtype=torch.float) / 25.0  # Normalize to [0, 1]

    # Generate text
    generated_text = generate_text(model, input_ids)

    # Model forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    reconstruction_loss = outputs.loss

    # Use only the first 'len(secret_message)' logits
    last_hidden_state = outputs.logits[:, -1, :26]  # Only use first 26 logits (A-Z)
    stego_prediction = torch.sigmoid(last_hidden_state[:, :len(secret_message)])

    # Overseer forward pass
    with torch.no_grad():
        overseer_outputs = overseer(input_ids=input_ids, attention_mask=attention_mask)
    overseer_last_hidden_state = overseer_outputs.logits[:, -1, :26]
    overseer_prediction = torch.sigmoid(overseer_last_hidden_state[:, :len(secret_message)])

    # Compute losses
    stego_loss = torch.nn.functional.mse_loss(stego_prediction, secret_tensor.unsqueeze(0))
    overseer_loss = torch.nn.functional.binary_cross_entropy(overseer_prediction, (stego_prediction > 0.5).float())

    # Adversarial reward
    overseer_success = ((overseer_prediction > 0.5) == (secret_tensor > 0.5)).float().mean()
    adversarial_reward = 1.0 - overseer_success

    # Total loss for the main model
    total_loss = reconstruction_loss + adversarial_reward * stego_loss

    # Backward pass and optimize
    model_optimizer.zero_grad()
    total_loss.backward()
    model_optimizer.step()

    # Train overseer
    overseer_optimizer.zero_grad()
    overseer_loss.backward()
    overseer_optimizer.step()

    # Decode the steganography prediction
    decoded_secret = decode_secret((stego_prediction.squeeze() * 25).round().int().tolist())

    return total_loss.item(), reconstruction_loss.item(), stego_loss.item(), overseer_loss.item(), generated_text, decoded_secret, overseer_success.item()
# Main execution
if __name__ == "__main__":
    num_epochs = 5000
    prompts = ["The weather is", "I like to", "My favorite"]
    secret_messages = ["HELLO", "WORLD", "SECRET"]

    for epoch in range(num_epochs):
        prompt = random.choice(prompts)
        secret_message = random.choice(secret_messages)

        total_loss, recon_loss, stego_loss, overseer_loss, generated_text, decoded_secret, overseer_success = train_step(prompt, secret_message)

        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Prompt: {prompt}')
            print(f'Secret message: {secret_message}')
            print(f'Generated text: {generated_text}')
            print(f'Decoded secret: {decoded_secret}')
            print(f'Total Loss: {total_loss:.4f}, Reconstruction Loss: {recon_loss:.4f}, Stego Loss: {stego_loss:.4f}')
            print(f'Overseer Loss: {overseer_loss:.4f}, Overseer Success Rate: {overseer_success:.4f}\n')

    # Test the model
    test_prompt = "The weather is"
    test_secret = "TEST"
    _, _, _, _, generated_text, decoded_secret, overseer_success = train_step(test_prompt, test_secret)
    print("Final Test:")
    print(f'Prompt: {test_prompt}')
    print(f'Secret message: {test_secret}')
    print(f'Generated text: {generated_text}')
    print(f'Decoded secret: {decoded_secret}')
    print(f'Overseer Success Rate: {overseer_success:.4f}')