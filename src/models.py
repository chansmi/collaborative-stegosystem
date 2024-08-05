import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead
import torch

def create_model(config):
    quantization_config = None
    if config['model'].get('quantize', False):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if 'NVIDIA' in gpu_name:
                # Use BitsAndBytesConfig for NVIDIA GPUs
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                print("Using BitsAndBytesConfig for NVIDIA GPU")
            else:
                # Use dynamic quantization for AMD GPUs
                quantization_config = None
                print("Using dynamic quantization for AMD GPU")
        else:
            # No GPU available, apply CPU quantization
            quantization_config = None
            print("Using dynamic quantization for CPU")

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply dynamic quantization if not using BitsAndBytesConfig
    if not quantization_config and config['model'].get('quantize', False):
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    # Wrap the model with AutoModelForCausalLMWithValueHead
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

    # Disable caching for training
    model.config.use_cache = False

    # Load and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def create_agent(config, role):
    model, tokenizer = create_model(config)
    return {
        'model': model,
        'tokenizer': tokenizer,
        'role': role
    }

def extract_decision(response):
    # Basic NLP technique to extract stock and direction
    stock_pattern = r'\b(AAPL|GOOGL|MSFT|AMZN|FB)\b'
    direction_pattern = r'\b(up|down)\b'
    
    stock_match = re.search(stock_pattern, response, re.IGNORECASE)
    direction_match = re.search(direction_pattern, response, re.IGNORECASE)
    
    stock = stock_match.group(0) if stock_match else None
    direction = direction_match.group(0) if direction_match else None
    
    return {'stock': stock, 'direction': direction}