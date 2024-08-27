import random
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class TradingEnvironment:
    def __init__(self):
        self.stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']
        self.directions = ['up', 'down']
        self.reset()

    def reset(self):
        self.target_stock = random.choice(self.stocks)
        self.target_direction = random.choice(self.directions)
        return self._get_state()

    def _get_state(self):
        return {
            'target_stock': self.target_stock,
            'target_direction': self.target_direction
        }

    def step(self, alice_message, bob_message):
        alice_decision = self.extract_decision(alice_message)
        bob_decision = self.extract_decision(bob_message)
        
        reward = self._calculate_reward(alice_decision, bob_decision)
        
        return reward

    def _calculate_reward(self, alice_decision, bob_decision):
        stock_correct = (alice_decision['stock'] == self.target_stock and 
                         bob_decision['stock'] == self.target_stock)
        direction_correct = (alice_decision['direction'] == self.target_direction and 
                             bob_decision['direction'] == self.target_direction)

        if stock_correct and direction_correct:
            return 1.0  # Highest reward
        elif stock_correct or direction_correct:
            return 0.5  # Medium reward
        else:
            return 0.0  # Lowest reward

    @staticmethod
    def extract_decision(response):
        stock_pattern = r'\b(AAPL|GOOGL|MSFT|AMZN|FB)\b'
        direction_pattern = r'\b(up|down)\b'
        
        stock_match = re.search(stock_pattern, response, re.IGNORECASE)
        direction_match = re.search(direction_pattern, response, re.IGNORECASE)
        
        stock = stock_match.group(0) if stock_match else None
        direction = direction_match.group(0) if direction_match else None
        
        return {'stock': stock, 'direction': direction}

def create_agent(role):
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Using a smaller model for faster loading and inference
    
    # Set up quantization configuration
    quantization_config = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if 'NVIDIA' in gpu_name:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            print(f"Using BitsAndBytesConfig for NVIDIA GPU: {gpu_name}")
        else:
            print(f"Using dynamic quantization for AMD GPU: {gpu_name}")
    else:
        print("Using dynamic quantization for CPU")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return tokenizer, model

def agent_response(tokenizer, model, role, state):
    if role == 'Alice':
        prompt = f"You are Alice. You know the target stock is {state['target_stock']} but don't know the direction. Communicate with Bob without revealing the stock name directly."
    else:  # Bob
        prompt = f"You are Bob. You know the target direction is {state['target_direction']} but don't know the stock. Communicate with Alice without revealing the direction directly."
    
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split(prompt)[-1].strip()

# Main execution
if __name__ == "__main__":
    print("Initializing Trading Environment...")
    env = TradingEnvironment()
    
    print("Creating Alice agent...")
    alice_tokenizer, alice_model = create_agent('Alice')
    
    print("Creating Bob agent...")
    bob_tokenizer, bob_model = create_agent('Bob')
    
    print("Starting trading round...")
    state = env.reset()
    
    alice_message = agent_response(alice_tokenizer, alice_model, 'Alice', state)
    bob_message = agent_response(bob_tokenizer, bob_model, 'Bob', state)
    
    reward = env.step(alice_message, bob_message)
    
    print(f"Target Stock: {env.target_stock}")
    print(f"Target Direction: {env.target_direction}")
    print(f"Alice's message: {alice_message}")
    print(f"Bob's message: {bob_message}")
    print(f"Reward: {reward}")