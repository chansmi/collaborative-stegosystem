from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import AutoModelForCausalLMWithValueHead
import torch

def is_rocm_available():
    # Check if ROCm is available
    try:
        import torch
        return torch.version.hip is not None
    except ImportError:
        return False

def create_model(config):
    if is_rocm_available():
        print("ROCm is available. Using ROCm compatible configuration.")
        # ROCm does not support bitsandbytes, so we will avoid using it
        quantization_config = None
    else:
        print("ROCm is not available. Using CUDA compatible configuration.")
        # Initialize quantization config for CUDA
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=config['model']['load_in_8bit'],
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

    # Load the base model with or without quantization
    if quantization_config:
        base_model = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            device_map="auto",
            trust_remote_code=True,
        )

    # Apply PEFT if enabled
    if config['model']['use_peft']:
        peft_config = LoraConfig(
            r=config['model']['lora_r'],
            lora_alpha=config['model']['lora_alpha'],
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        base_model = get_peft_model(base_model, peft_config)

    # Wrap the model with AutoModelForCausalLMWithValueHead
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)

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