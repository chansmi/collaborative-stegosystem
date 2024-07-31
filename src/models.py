# src/models.py

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import AutoModelForCausalLMWithValueHead

def create_model(config):
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=config['model']['load_in_8bit'],
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
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

    # Wrap the model with AutoModelForCausalLMWithValueHead
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

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