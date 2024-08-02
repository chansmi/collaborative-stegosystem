from trl import SFTTrainer
from transformers import TrainingArguments
from models import create_model

def train_sft(config, dataset):
    model, tokenizer = create_model(config)

    training_args = TrainingArguments(
        output_dir=config['sft']['output_dir'],
        num_train_epochs=config['sft']['num_train_epochs'],
        per_device_train_batch_size=config['sft']['batch_size'],
        learning_rate=config['sft']['learning_rate'],
        fp16=True,
        gradient_checkpointing=True,
        save_steps=config['sft']['save_steps'],
        logging_steps=config['sft']['logging_steps'],
        save_total_limit=config['sft']['save_total_limit'],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(config['sft']['output_dir'])

    return model, tokenizer

# Train SFT models for Alice and Bob
alice_sft_model, alice_tokenizer = train_sft(config['alice'], alice_dataset)
bob_sft_model, bob_tokenizer = train_sft(config['bob'], bob_dataset)