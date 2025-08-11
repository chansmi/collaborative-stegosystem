#!/usr/bin/env python3
"""
Cleanup script for the experiments directory.
Removes broken, duplicate, or unnecessary experiment files.
"""

import os
import shutil
from pathlib import Path

def cleanup_experiments():
    """Clean up the experiments directory."""
    experiments_dir = Path("experiments")
    
    # Files to remove (broken, duplicate, or not working)
    files_to_remove = [
        "tioga12-pt_main_thread-2929517.core",  # Core dump file
        "Untitled.ipynb",  # Unnamed notebook
        "gpt2-sentiment.ipynb",  # Large notebook that may not be needed
        ".ipynb_checkpoints",  # Jupyter checkpoint directory
    ]
    
    # Remove specified files and directories
    for item in files_to_remove:
        item_path = experiments_dir / item
        if item_path.exists():
            if item_path.is_file():
                item_path.unlink()
                print(f"Removed file: {item}")
            elif item_path.is_dir():
                shutil.rmtree(item_path)
                print(f"Removed directory: {item}")
    
    # Keep only the most important experiment files
    # These represent different approaches and are working
    important_experiments = [
        "1_experiment.py",           # Basic experiment
        "5_encoder_decoder.py",      # Encoder-decoder approach
        "10_gpt_overseer.py",        # GPT overseer
        "11_gpt_overseer_2.py",      # GPT overseer variant
        "14_improved_learning.py",   # Improved learning
        "15_movie_review.py",        # Movie review domain
        "18b_simple_ppo.py",         # Simple PPO
        "20_segmeneted_ppo.py",      # Segmented PPO
        "21_own_ppo.py",             # Custom PPO
        "22_trial_trl.py",           # TRL trial
        "23_trl_wandb_log.py",       # TRL with wandb
        "24_adjusted_model.py",      # Adjusted model
        "25_ppo_validation.py",      # PPO validation
        "26_ppo_new_prompt.py",      # PPO with new prompts
    ]
    
    # List all Python files in experiments
    all_py_files = list(experiments_dir.glob("*.py"))
    
    # Remove files that are not in the important list
    for py_file in all_py_files:
        if py_file.name not in important_experiments:
            print(f"Removing less important experiment: {py_file.name}")
            py_file.unlink()
    
    print(f"\nCleanup completed. Kept {len(important_experiments)} important experiments.")
    print("Remaining experiments represent different steganography approaches.")

if __name__ == "__main__":
    print("Cleaning up experiments directory...")
    cleanup_experiments()
    print("Done!")
