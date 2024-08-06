# models.py
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead
import torch

def create_agent(config, role):
    quantization_config = None
    if config['model'].get('quantize', False):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if 'NVIDIA' in gpu_name:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                print("Using BitsAndBytesConfig for NVIDIA GPU")
            else:
                print("Using dynamic quantization for AMD GPU")
        else:
            print("Using dynamic quantization for CPU")

    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )


    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512  # or any other value you prefer

    return {
        'model': model,
        'tokenizer': tokenizer,
        'role': role
    }

def extract_decision(response):
    stock_pattern = r'\b(AAPL|GOOGL|MSFT|AMZN|FB)\b'
    direction_pattern = r'\b(up|down)\b'
    
    stock_match = re.search(stock_pattern, response, re.IGNORECASE)
    direction_match = re.search(direction_pattern, response, re.IGNORECASE)
    
    stock = stock_match.group(0) if stock_match else None
    direction = direction_match.group(0) if direction_match else None
    
    return {'stock': stock, 'direction': direction}