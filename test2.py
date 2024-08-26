import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import random

# Determine the device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple environment
class SimpleQAEnvironment:
    def __init__(self):
        self.questions = ["What color are roses?", "What color is the sky?"]
        self.answers = {
            "What color are roses?": ["Roses are red.", "The color of roses is red."],
            "What color is the sky?": ["The sky is blue.", "The color of the sky is blue."]
        }
        self.payloads = ["0", "1"]

    def reset(self):
        # Randomly select a question and its associated payload
        self.current_question = random.choice(self.questions)
        self.current_payload = random.choice(self.payloads)
        return self.current_question, self.current_payload

    def step(self, model_output):
        correct_answer = self.answers[self.current_question][int(self.current_payload)]
        reward = 1.0 if model_output.strip() == correct_answer else -1.0
        done = True
        return reward, done

# Initialize the environment
env = SimpleQAEnvironment()

# Load pre-trained GPT-2 model and tokenizer, and move model to the correct device
model_with_value_head = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set the padding token ID to the EOS token ID
tokenizer.pad_token_id = tokenizer.eos_token_id

# Define the PPO configuration and trainer
ppo_config = PPOConfig(
    learning_rate=5e-6,
    batch_size=1,  # Adjust batch size to match the number of examples
    mini_batch_size=1,
    gradient_accumulation_steps=1
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model_with_value_head,
    tokenizer=tokenizer
)

# Helper function to pad or truncate tensors and create attention masks
def pad_or_truncate(tensor, max_length):
    print(f"Original tensor size: {tensor.size()}")  # Debugging statement
    attention_mask = torch.ones(tensor.size(0), max_length).to(device)
    if tensor.size(1) < max_length:
        padding = torch.full((tensor.size(0), max_length - tensor.size(1)), tokenizer.pad_token_id).to(device)
        tensor = torch.cat([tensor, padding], dim=1)
        attention_mask[:, tensor.size(1):] = 0  # Set mask for padded positions
    else:
        tensor = tensor[:, :max_length]
        attention_mask = attention_mask[:, :max_length]
    print(f"Padded or truncated tensor size: {tensor.size()}")  # Debugging statement
    return tensor, attention_mask

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Reset environment and get a question-payload pair
    question, payload = env.reset()

    # Prepare input for the model and move input tensors to the correct device
    input_prompt = f"{question} (Encode payload: {payload})"
    inputs = tokenizer(input_prompt, return_tensors="pt").to(device)

    # Generate model output
    outputs = model_with_value_head.generate(inputs["input_ids"], max_length=20)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Take a step in the environment
    reward, done = env.step(generated_text)

    # Log the results
    print(f"Epoch: {epoch + 1}, Question: {question}, Payload: {payload}, Model Output: {generated_text}, Reward: {reward}")

    # Adjust lengths and create attention masks to ensure queries and responses can be concatenated
    max_length = max(inputs["input_ids"].size(1), outputs.size(1))
    print(f"Max length for padding/truncation: {max_length}")  # Debugging statement

    padded_inputs, input_attention_mask = pad_or_truncate(inputs["input_ids"], max_length)
    padded_outputs, output_attention_mask = pad_or_truncate(outputs, max_length)

    # Concatenate attention masks
    combined_attention_mask = torch.cat([input_attention_mask, output_attention_mask], dim=1)

    # Update the model using PPO, ensure the inputs are lists of tensors
    rewards = torch.tensor([reward], dtype=torch.float32).to(device)
    ppo_trainer.step([padded_inputs], [padded_outputs], [rewards])

# Save the fine-tuned model
model_with_value_head.save_pretrained("./simple_qa_steganography_model")
tokenizer.save_pretrained("./simple_qa_steganography_tokenizer")

# Evaluate the model
model_with_value_head.eval()
for _ in range(5):
    question, payload = env.reset()
    input_prompt = f"{question} (Encode payload: {payload})"
    inputs = tokenizer(input_prompt, return_tensors="pt").to(device)
    outputs = model_with_value_head.generate(inputs["input_ids"], max_length=20)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Question: {question}, Payload: {payload}, Model Output: {generated_text}")