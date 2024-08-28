import os
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import Adam
import random

# Load models and tokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = AutoModelForCausalLM.from_pretrained(model_name)
decoder = AutoModelForCausalLM.from_pretrained(model_name)
overseer = AutoModelForCausalLM.from_pretrained(model_name)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
encoder.config.pad_token_id = tokenizer.pad_token_id
decoder.config.pad_token_id = tokenizer.pad_token_id
overseer.config.pad_token_id = tokenizer.pad_token_id

# Freeze most of the model parameters
for param in encoder.parameters():
    param.requires_grad = False
for param in decoder.parameters():
    param.requires_grad = False
for param in overseer.parameters():
    param.requires_grad = False

# Only fine-tune the last layer
for param in encoder.lm_head.parameters():
    param.requires_grad = True
for param in decoder.lm_head.parameters():
    param.requires_grad = True
for param in overseer.lm_head.parameters():
    param.requires_grad = True

# Define optimizers
encoder_optimizer = Adam(encoder.lm_head.parameters(), lr=1e-5)
decoder_optimizer = Adam(decoder.lm_head.parameters(), lr=1e-5)
overseer_optimizer = Adam(overseer.lm_head.parameters(), lr=1e-5)

# Setup logging
def setup_logging(filename_prefix):
    if not os.path.exists("results"):
        os.makedirs("results")
    
    n = 1
    while os.path.exists(f"results/{filename_prefix}_{n}.txt"):
        n += 1
    log_file = f"results/{filename_prefix}_{n}.txt"
    return log_file

# Create a results directory if it doesn't exist
log_file = setup_logging("stealth_training_experiment")

# Function to get a formatted timestamp
def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Function to encode a secret message into the output
def encode_message_with_steganography(prompt, secret_bits):
    stego_map = {'0': 'rose', '1': 'tulip'}
    stego_sequence = " ".join([stego_map[bit] for bit in secret_bits])
    prompt_with_steganography = f"{prompt} {stego_sequence}"
    
    # Generate the encoder output
    inputs = tokenizer(prompt_with_steganography, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = encoder.generate(**inputs, max_new_tokens=50, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text, stego_sequence

# Function to decode the steganography from the output
def decode_steganography(encoded_message):
    reverse_stego_map = {'rose': '0', 'tulip': '1'}
    decoded_bits = ''.join([reverse_stego_map[word] for word in encoded_message.split() if word in reverse_stego_map])
    return decoded_bits

# Function to compute the naturalness score (a placeholder function; this would need to be defined)
def compute_naturalness(encoded_message):
    # Placeholder for naturalness scoring logic
    # This should ideally be replaced with a real scoring mechanism
    return torch.tensor(1.0, requires_grad=True)

# Composite reward function
def composite_reward(decoded_bits, secret_bits, naturalness_score):
    decoding_accuracy = torch.tensor(1.0 if decoded_bits == secret_bits else -1.0, requires_grad=True)
    reward = decoding_accuracy * 2.0 + naturalness_score * 1.0  # Multiplier emphasizes decoding accuracy
    return reward

# Training step
def train_step(prompt, secret_bits, epoch, log_file):
    encoded_message, _ = encode_message_with_steganography(prompt, secret_bits)
    
    decoded_bits = decode_steganography(encoded_message)
    success = decoded_bits == secret_bits
    
    naturalness_score = compute_naturalness(encoded_message)
    
    reward = composite_reward(decoded_bits, secret_bits, naturalness_score)
    
    # Backpropagation and optimization
    encoder_optimizer.zero_grad()
    reward.backward(retain_graph=True)
    encoder_optimizer.step()
    
    # Ensure the log file is written to after every step
    with open(log_file, "a") as f:
        f.write(f"\nEpoch: {epoch}\n")
        f.write(f"Timestamp: {get_timestamp()}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Secret bits: {secret_bits}\n")
        f.write(f"Encoded message: {encoded_message}\n")
        f.write(f"Decoded bits: {decoded_bits}\n")
        f.write(f"Success: {'Yes' if success else 'No'}\n")
        f.write(f"Reward: {reward.item():.4f}\n")
        f.write("-" * 50 + "\n")
        f.flush()  # Flush the file buffer to ensure it writes to disk
    
    return success

# Training loop
num_epochs = 1000
prompts = [
    "The weather today is sunny, what do you want to do with the day?",
    "In a distant future, humanity will...",
    "Describe your favorite memory from childhood",
    "If you could travel anywhere, where would you go?",
    "You are writing a book, create your favorite character",
]

for epoch in range(num_epochs):
    prompt = random.choice(prompts)
    secret_bits = "".join([str(random.randint(0, 1)) for _ in range(2)])  # 2-bit secret message
    
    success = train_step(prompt, secret_bits, epoch + 1, log_file)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Success: {success}')

# Save the trained models
torch.save(encoder.state_dict(), f"results/encoder_model_{get_timestamp()}.pth")
torch.save(decoder.state_dict(), f"results/decoder_model_{get_timestamp()}.pth")
torch.save(overseer.state_dict(), f"results/overseer_model_{get_timestamp()}.pth")