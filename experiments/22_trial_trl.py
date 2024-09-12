from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator, PartialState
from datasets import load_from_disk
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

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
    return load_from_disk('./collaborative-stegosystem/data/simple_ppo_data')

# Load dataset
with PartialState().local_main_process_first():
    dataset = build_dataset(ppo_config.query_dataset, ppo_config.dataset_num_proc)

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

# Set seed for reproducibility
set_seed(ppo_config.seed)

# Initialize models
if not args.use_peft:
    ref_model = trl_model_class.from_pretrained(ppo_config.model_name, trust_remote_code=args.trust_remote_code)
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
sender_model = trl_model_class.from_pretrained(
    ppo_config.model_name,
    trust_remote_code=args.trust_remote_code,
    device_map=device_map,
    peft_config=peft_config,
)
receiver_model = trl_model_class.from_pretrained(
    ppo_config.model_name,
    trust_remote_code=args.trust_remote_code,
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
    "max_new_tokens": 20,
}

# Training loop
for batch in tqdm(sender_ppo_trainer.dataloader):
    query_tensors = batch["input_ids"]

    # Generate responses
    response_tensors, ref_response_tensors = sender_ppo_trainer.generate(
        query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
    )
    batch["sender_response"] = tokenizer.batch_decode(response_tensors)
    batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

    # Prepare receiver queries
    receiver_queries = [f"Color: {response}, Guess the payload:" for response in batch["sender_response"]]
    tokenized_receiver_inputs = tokenizer(receiver_queries, return_tensors="pt", padding=True, truncation=True)
    query_tensors_receiver = tokenized_receiver_inputs.input_ids.to(receiver_model.pretrained_model.device)
    
    # Generate receiver responses
    query_tensor_receiver_list = [query_tensors_receiver[i] for i in range(len(query_tensors_receiver))]
    receiver_response = receiver_ppo_trainer.generate(query_tensor_receiver_list, generate_ref_response=False, **generation_kwargs)
    batch["receiver_response"] = tokenizer.batch_decode(receiver_response)

    # Compute rewards
    rewards = []
    for original_payload, receiver_response in zip(batch["label"], batch["receiver_response"]):
        try:
            guessed_payload = int(receiver_response.strip())
            reward = 1.0 if guessed_payload == original_payload else -1.0
            reward += 0.1 if ('green' in original_payload or 'red' in original_payload) else 0.0
        except ValueError:
            reward = -1.0  # Invalid response
        rewards.append(torch.tensor(reward).to(receiver_model.pretrained_model.device))

    # Run PPO step
    sender_stats = sender_ppo_trainer.step(query_tensors, response_tensors, rewards)
    receiver_stats = receiver_ppo_trainer.step(query_tensors, response_tensors, rewards)