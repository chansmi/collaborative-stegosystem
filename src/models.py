# models.py
import re
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead
import torch

def create_agent(config, role):
    """Create an agent with robust model loading and error handling."""
    
    # Check if we have the required authentication
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set. Please set it to access Meta-Llama models.")
    
    model_name = config['model']['name']
    
    # Set up quantization config based on hardware
    quantization_config = None
    device_map = "auto"
    
    if config['model'].get('quantize', False):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if 'NVIDIA' in gpu_name:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                    )
                    print(f"Using 8-bit quantization for NVIDIA GPU: {gpu_name}")
                except Exception as e:
                    print(f"8-bit quantization failed: {e}, falling back to 16-bit")
                    quantization_config = None
            else:
                print(f"Using dynamic quantization for GPU: {gpu_name}")
                quantization_config = None
        else:
            print("Using dynamic quantization for CPU")
            quantization_config = None
            device_map = "cpu"
    
    # Try to load the model with fallbacks
    try:
        # First try with flash attention 2
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if quantization_config is None else None,
            attn_implementation="flash_attention_2",
            token=hf_token,
        )
        print("Successfully loaded model with flash_attention_2")
    except Exception as e:
        print(f"Flash attention 2 failed: {e}, trying without it")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16 if quantization_config is None else None,
                token=hf_token,
            )
            print("Successfully loaded model without flash_attention_2")
        except Exception as e2:
            print(f"Model loading failed: {e2}")
            raise RuntimeError(f"Failed to load model {model_name}: {e2}")

    # Convert to PPO model
    try:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        print(f"Successfully converted {role} model to PPO format")
    except Exception as e:
        print(f"Failed to convert to PPO model: {e}")
        raise RuntimeError(f"PPO conversion failed for {role}: {e}")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = 512
        
        # Ensure tokenizer is on the same device as model
        if hasattr(model, 'device'):
            tokenizer.device = model.device
        print(f"Successfully loaded tokenizer for {role}")
    except Exception as e:
        print(f"Tokenizer loading failed: {e}")
        raise RuntimeError(f"Tokenizer loading failed for {role}: {e}")

    return {
        'model': model,
        'tokenizer': tokenizer,
        'role': role
    }

def extract_decision(response):
    """Extract stock and direction decisions from agent responses."""
    stock_pattern = r'\b(AAPL|GOOGL|MSFT|AMZN|FB)\b'
    direction_pattern = r'\b(up|down)\b'
    
    stock_match = re.search(stock_pattern, response, re.IGNORECASE)
    direction_match = re.search(direction_pattern, response, re.IGNORECASE)
    
    stock = stock_match.group(0) if stock_match else None
    direction = direction_match.group(0) if direction_match else None
    
    return {'stock': stock, 'direction': direction}