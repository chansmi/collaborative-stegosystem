from trl import RewardTrainer
from transformers import TrainingArguments
from models import create_model

def train_reward_model(config, dataset):
    model, tokenizer = create_model(config)

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

    return model, tokenizer

# Train Reward model
reward_model, reward_tokenizer = train_reward_model(config['reward'], reward_dataset)