from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def create_model(config):
    model = AutoModelForCausalLM.from_pretrained(config['model']['name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

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

    return model, tokenizer

def create_agent(config, role):
    model, tokenizer = create_model(config)
    return {
        'model': model,
        'tokenizer': tokenizer,
        'role': role
    }