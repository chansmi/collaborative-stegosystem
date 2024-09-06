from datasets import Dataset
import logging
import random
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Define colors and payloads    
COLORS = ["red", "green", "blue", "yellow"]
PAYLOADS = [0, 1]

# Create dataset
def create_dataset(num_samples=1000):
    logger.info(f"Creating dataset with {num_samples} samples...")
    data = []
    for _ in range(num_samples):
        payload = random.choice(PAYLOADS)
        data.append({
            "label": payload,
            "query": f"Payload: {payload}, Choose a color out of red, blue, green, or yellow:"
        })
    logger.info(f"Dataset created with {len(data)} samples")

    ds = Dataset.from_list(data)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"])
        return sample

    ds = ds.map(tokenize)
    ds.set_format(type="torch")
    return ds

# Generate the dataset
dataset = create_dataset()

# Save the dataset to a file in the current directory
output_dir = "./collaborative-stegosystem/data/simple_ppo_data"
logger.info(f"Saving dataset to {os.path.abspath(output_dir)}...")
dataset.save_to_disk(output_dir)
logger.info("Dataset saved successfully.")