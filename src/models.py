from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model

def create_model(config):
    # Load the base model with AutoModelForCausalLMWithValueHead
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config['model']['name'],
        load_in_8bit=config['model']['load_in_8bit'],
        device_map="auto",
        trust_remote_code=True
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
        model = get_peft_model(model, peft_config)

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