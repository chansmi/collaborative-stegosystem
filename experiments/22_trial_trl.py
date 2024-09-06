# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python examples/scripts/ppo.py \
    --log_with=wandb
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator, PartialState
from datasets import load_from_disk
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available
import os
import json
from datetime import datetime


tqdm.pandas()


@dataclass
class ScriptArguments:
    use_seq2seq: bool = field(default=False, metadata={"help": "whether to use seq2seq"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})

    # LoraConfig
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})


parser = HfArgumentParser((ScriptArguments, PPOConfig))
args, ppo_config = parser.parse_args_into_dataclasses()

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 8}

trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead

tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(query_dataset, dataset_num_proc, input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        query_dataset (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    # load imdb with datasets

    ds = load_from_disk('./collaborative-stegosystem/data/simple_ppo_data')
    return ds

#["label", "query", "response"] weird ppo thing



    return new_file_path

# We retrieve the dataloader by calling the `build_dataset` function.
# Compute that only on the main process for faster data processing.
# see: https://github.com/huggingface/trl/pull/1255
with PartialState().local_main_process_first():
    dataset = build_dataset(ppo_config.query_dataset, ppo_config.dataset_num_proc)


def collator(data):
    #return {key: [d[key] for d in data] for key in data[0]}
    out = {key: [d[key] for d in data] for key in data[0]}
    return out


# set seed before initializing value head for deterministic eval
set_seed(ppo_config.seed)

# Now let's build the model, the reference model, and the tokenizer.
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
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}

# Sender
sender_model = trl_model_class.from_pretrained(
    ppo_config.model_name,
    trust_remote_code=args.trust_remote_code,
    device_map=device_map,
    peft_config=peft_config,
)
#Recciever 
receiver_model = trl_model_class.from_pretrained(
    ppo_config.model_name,
    trust_remote_code=args.trust_remote_code,
    device_map=device_map,
    peft_config=peft_config,
)


#tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
5
# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
#tokenizer.pad_token_id = tokenizer.eos_token_id

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
sender_ppo_trainer = PPOTrainer(ppo_config, sender_model, ref_model, tokenizer, dataset=dataset, data_collator=collator, log_with="wandb")
receiver_ppo_trainer = PPOTrainer(ppo_config, receiver_model, ref_model, tokenizer, log_with="wandb")

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 1,
}


for batch in tqdm(sender_ppo_trainer.dataloader):
    query_tensors = batch["input_ids"]

    # Get response from gpt2
    response_tensors, ref_response_tensors = sender_ppo_trainer.generate(
        query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
    )
    batch["sender_response"] = tokenizer.batch_decode(response_tensors)
    batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

# Tokenize and pad receiver queries
    receiver_queries = [f"Color: {response}, Guess the payload:" for response in batch["sender_response"]]

    # Tokenize the receiver queries
    tokenized_receiver_inputs = tokenizer(receiver_queries, return_tensors="pt", padding=True, truncation=True)
    query_tensors_receiver = tokenized_receiver_inputs.input_ids.to(receiver_model.pretrained_model.device)
    # Generate responses from the receiver model
    query_tensor_receiver_list = [query_tensors_receiver[i] for i in range(len(query_tensors_receiver))]
    receiver_response = receiver_ppo_trainer.generate(query_tensor_receiver_list, generate_ref_response=False, **generation_kwargs)

    batch["receiver_response"] = tokenizer.batch_decode(receiver_response)

    # Compute sentiment score
    # this is correct when we have a receiver model
    rewards = []
    for original_payload, receiver_response in zip(batch["label"], batch["receiver_response"]):
        try:
            guessed_payload = int(receiver_response.strip())
            reward = 1.0 if guessed_payload == original_payload else -1.0
            reward = reward + 0.1 if ('green' in original_payload or 'red' in original_payload) else 0.0
        except ValueError:
            reward = -1.0  # Invalid response
        rewards.append(torch.tensor(reward).to(receiver_model.pretrained_model.device))

    # rewards = []
    # for sender_response in batch["response"]:
    #     reward = 1.0 if 'green' in sender_response else 0.0
    #     rewards.append(torch.tensor(reward).to(sender_model.pretrained_model.device))
    # Run PPO step
    sender_stats = sender_ppo_trainer.step(query_tensors, response_tensors, rewards)
    receiver_stats = receiver_ppo_trainer.step(query_tensors, response_tensors, rewards)

    sender_ppo_trainer.log_stats(sender_model, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])
    #print(batch["response"])
