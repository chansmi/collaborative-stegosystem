# train_reward.py
from trl import RewardTrainer
from transformers import TrainingArguments

def train_reward_model(config, model, tokenizer, dataset):
    training_args = TrainingArguments(
        output_dir=config['reward']['output_dir'],
        num_train_epochs=config['reward']['num_train_epochs'],
        per_device_train_batch_size=config['reward']['batch_size'],
        learning_rate=config['reward']['learning_rate'],
        fp16=True,
        gradient_checkpointing=True,
        save_steps=config['reward']['save_steps'],
        logging_steps=config['reward']['logging_steps'],
        save_total_limit=config['reward']['save_total_limit'],
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(config['reward']['output_dir'])

    return trainer.model