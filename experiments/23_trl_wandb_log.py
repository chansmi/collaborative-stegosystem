from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator, PartialState
from datasets import load_from_disk
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
import wandb

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed

tqdm.pandas()

@dataclass
class ScriptArguments:
    use_seq2seq: bool = field(default=False, metadata={"help": "Whether to use seq2seq"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    use_peft: bool = field(default=False, metadata={"help": "Whether to use PEFT"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "The LoRA alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "The LoRA r parameter"})

# Parse arguments
parser = HfArgumentParser((ScriptArguments, PPOConfig))
args, ppo_config = parser.parse_args_into_dataclasses()

# Set up model class and tokenizer
trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead
tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

def build_dataset(query_dataset, dataset_num_proc, input_min_text_length=2, input_max_text_length=8):
    return load_from_disk('/g/g10/smith585/codebases/collaborative-stegosystem/data/simple_ppo_data')

# Load dataset
with PartialState().local_main_process_first():
    dataset = build_dataset(ppo_config.query_dataset, ppo_config.dataset_num_proc)

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


# Initial PPO configuration
ppo_config = PPOConfig()

# Update PPO configuration
ppo_config.ppo_epochs = 10  
ppo_config.batch_size = 64  
ppo_config.mini_batch_size = 16  
num_batches_per_epoch = len(dataset) // ppo_config.batch_size
ppo_config.steps = ppo_config.ppo_epochs * num_batches_per_epoch

# PPO Configuration
ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=5e-6,  # Reduced learning rate for stability
    log_with="wandb",
    batch_size=ppo_config.batch_size,
    mini_batch_size=ppo_config.mini_batch_size,
    gradient_accumulation_steps=2,  # Increased for more stable updates
    optimize_device_cache=True,
    early_stopping=False,
    target_kl=0.1,  # Reduced target KL for more conservative updates
    ppo_epochs=ppo_config.ppo_epochs,
    max_grad_norm=1.0,  # Added gradient clipping
    seed=42,
    init_kl_coef=0.2,
    adap_kl_ctrl=True,
    cliprange=0.1,  # Reduced cliprange for more conservative updates
    cliprange_value=0.1,  # Reduced cliprange_value for more conservative updates
    vf_coef=0.1,
    horizon=10000,
    gamma=0.99,  # Slightly reduced from 1.0 for more focus on immediate rewards
    lam=0.95,
    task_name="collaborative-stegosystem",
    tracker_project_name="collaborative-stegosystem",
    tracker_kwargs={"wandb": {"entity": "chansmi"}},
    steps=ppo_config.steps,
)

print(f"Total steps: {ppo_config.steps}")
print(f"Number of epochs: {ppo_config.ppo_epochs}")
print(f"Batches per epoch: {num_batches_per_epoch}")
print(f"Batch size: {ppo_config.batch_size}")
print(f"Mini-batch size: {ppo_config.mini_batch_size}")
print(f"Total steps: {ppo_config.steps}")
print(f"Number of epochs: {ppo_config.ppo_epochs}")
print(f"Batches per epoch: {num_batches_per_epoch}")

# Initialize wandb
wandb.init(
    project=ppo_config.tracker_project_name,
    entity=ppo_config.tracker_kwargs["wandb"]["entity"],
    config=ppo_config.__dict__,
)

# Set up tokenizer
tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

# Set seed for reproducibility
set_seed(ppo_config.seed)

# Initialize models
if not args.use_peft:
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name)
    device_map = None
    peft_config = None
else:
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
    ref_model = None
    device_map = {"": Accelerator().local_process_index}

# Initialize sender and receiver models
sender_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    ppo_config.model_name,
    device_map=device_map,
    peft_config=peft_config,
)
receiver_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    ppo_config.model_name,
    device_map=device_map,
    peft_config=peft_config,
)

# Initialize PPO trainers
sender_ppo_trainer = PPOTrainer(ppo_config, sender_model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
receiver_ppo_trainer = PPOTrainer(ppo_config, receiver_model, ref_model, tokenizer)

# Define generation parameters
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 1,
}

# Training loop
train_total_correct = 0
train_total_samples = 0

for epoch in range(ppo_config.ppo_epochs):
    for batch in tqdm(sender_ppo_trainer.dataloader, desc=f"Epoch {epoch+1}/{ppo_config.ppo_epochs}"):
        query_tensors = batch["input_ids"]

        # Generate responses
        response_tensors, ref_response_tensors = sender_ppo_trainer.generate(
            query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
        )
        batch["sender_response"] = tokenizer.batch_decode(response_tensors)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

        # Prepare receiver queries
        receiver_queries = [f"Color: {response}, Guess the payload:" for response in batch["sender_response"]]
        tokenized_receiver_inputs = tokenizer(receiver_queries, return_tensors="pt", padding=True, truncation=True, max_length=512)
        query_tensors_receiver = tokenized_receiver_inputs.input_ids.to(receiver_model.pretrained_model.device)
        
        # Generate receiver responses
        query_tensor_receiver_list = [query_tensors_receiver[i] for i in range(len(query_tensors_receiver))]
        receiver_response = receiver_ppo_trainer.generate(query_tensor_receiver_list, generate_ref_response=False, **generation_kwargs)
        batch["receiver_response"] = tokenizer.batch_decode(receiver_response)

        print(batch["sender_response"])
        print(batch["receiver_response"])

        # Compute rewards and accuracy
        rewards = []
        for original_payload, receiver_response in zip(batch["label"], batch["receiver_response"]):
            try:
                guessed_payload = int(receiver_response.strip())
                reward = 1.0 if guessed_payload == original_payload else -1.0
                reward += 0.1 if ('green' in original_payload or 'red' in original_payload) else 0.0
                
                # Accuracy calculation
                if guessed_payload == original_payload:
                    train_total_correct += 1
                train_total_samples += 1
            except ValueError:
                reward = -1.0  # Invalid response
                train_total_samples += 1
            rewards.append(torch.tensor(reward).to(receiver_model.pretrained_model.device))

        # Run PPO step
        sender_stats = sender_ppo_trainer.step(query_tensors, response_tensors, rewards)
        receiver_stats = receiver_ppo_trainer.step(query_tensors, response_tensors, rewards)

        # Log stats
        current_train_accuracy = train_total_correct / train_total_samples if train_total_samples > 0 else 0
        wandb.log({
            "epoch": epoch,
            "sender_loss": sender_stats.get("loss", 0),
            "receiver_loss": receiver_stats.get("loss", 0),
            "sender_learning_rate": sender_stats.get("learning_rate", 0),
            "receiver_learning_rate": receiver_stats.get("learning_rate", 0),
            "batch_reward": sum(rewards).item() / len(rewards) if rewards else 0,
            "sender_entropy": sender_stats.get("entropy", 0),
            "receiver_entropy": receiver_stats.get("entropy", 0),
            "train_accuracy": current_train_accuracy,
        })

    print(f"Epoch {epoch+1}/{ppo_config.ppo_epochs} - Train Accuracy: {current_train_accuracy:.4f}")

# Final accuracy
final_train_accuracy = train_total_correct / train_total_samples if train_total_samples > 0 else 0
print(f"Final Train Accuracy: {final_train_accuracy:.4f}")
wandb.log({"final_train_accuracy": final_train_accuracy})

# Save the final models
sender_model.save_pretrained("./final_sender_model")
receiver_model.save_pretrained("./final_receiver_model")

wandb.finish()