# Project Structure Guide

This document explains the improved project structure for the Collaborative Stegosystem project.

## 🏗️ Overview

The new structure separates **shared resources** (models, data) from **project-specific outputs** (checkpoints, logs, results) while maintaining easy access to both.

## 📁 Directory Structure

```
/usr/workspace/smith585/
├── shared_models/                    # 🎯 CENTRAL LOCATION FOR ALL MODELS
│   ├── hf_cache/                     # Hugging Face cache
│   ├── transformers_cache/           # Transformers cache
│   ├── Meta-Llama-3.1-8B-Instruct/  # Your downloaded models
│   └── custom_models/               # Your trained models
├── shared_data/                      # Common datasets, embeddings
├── codebases/
│   └── collaborative-stegosystem/    # This project
│       ├── src/                      # Source code
│       ├── config/                   # Configuration files
│       ├── outputs/                  # 🎯 ALL PROJECT OUTPUTS
│       │   ├── checkpoints/          # Training checkpoints
│       │   ├── loras/               # LoRA adapters
│       │   ├── results/             # Experiment results
│       │   ├── logs/                # Training logs
│       │   └── models/              # Project-specific model outputs
│       ├── experiments/             # Experiment configurations
│       ├── data/                    # Project-specific data
│       └── shared_models -> ../shared_models  # Symbolic link
```

## 🔄 Migration from Old Structure

### Before (Old Structure):
```
codebases/collaborative-stegosystem/
├── models/           # Mixed with project outputs
├── checkpoints/      # Scattered
├── results/          # Scattered
├── logs/            # Scattered
└── src/
```

### After (New Structure):
```
codebases/collaborative-stegosystem/
├── outputs/          # All project outputs organized
│   ├── checkpoints/
│   ├── loras/
│   ├── results/
│   ├── logs/
│   └── models/
├── shared_models -> ../shared_models  # Symbolic link
└── src/
```

## 🚀 How to Use the New Structure

### 1. Import the Path Manager

```python
from src.paths import get_project_paths

paths = get_project_paths()
```

### 2. Access Different Paths

```python
# Get paths for different outputs
checkpoint_path = paths.get_checkpoint_path("ppo_training_001")
lora_path = paths.get_lora_path("alice_lora_v1")
model_path = paths.get_model_path("trained_alice", "final")
log_path = paths.get_log_path("training.log")

# Access shared resources
shared_model_path = paths.get_shared_model_path("Meta-Llama-3.1-8B-Instruct")
shared_data_path = paths.get_shared_data_path("stock_data")
```

### 3. Update Your Configuration

Use the new `config_improved.yaml` which has better organization:

```yaml
model:
  base_model: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  save_path: "outputs/models"  # Will be expanded automatically
  checkpoint_dir: "outputs/checkpoints"
  lora_dir: "outputs/loras"

ppo:
  output_dir: "outputs/ppo_results"
  checkpoint_dir: "outputs/checkpoints/ppo"
```

## 🎯 Benefits of the New Structure

### ✅ **Clear Separation**
- **Shared resources** (models, data) are centralized
- **Project outputs** (checkpoints, logs) are project-specific
- No more confusion about what belongs where

### ✅ **Easy Sharing**
- Models can be used across multiple projects
- No need to duplicate large model files
- Symbolic links provide easy access

### ✅ **Better Organization**
- All outputs are in one place (`outputs/`)
- Consistent naming conventions
- Easy to find and manage files

### ✅ **Scalability**
- Easy to add new projects
- Clear structure for new team members
- Better version control practices

## 🔧 Implementation Steps

### Step 1: Run the Migration Script
```bash
cd codebases/collaborative-stegosystem
python migrate_structure.py
```

### Step 2: Update Your Code
Replace hardcoded paths with the new path manager:

```python
# OLD WAY
model_path = "models/trained_model"

# NEW WAY
from src.paths import get_project_paths
paths = get_project_paths()
model_path = paths.get_model_path("trained_model")
```

### Step 3: Update Configuration Files
Use `config_improved.yaml` as a template for your configurations.

### Step 4: Test Everything
Make sure all your training scripts and evaluation code still work.

## 📋 Best Practices

### 1. **Always Use the Path Manager**
```python
# ✅ Good
paths = get_project_paths()
checkpoint_path = paths.get_checkpoint_path("experiment_001")

# ❌ Bad
checkpoint_path = "checkpoints/experiment_001"
```

### 2. **Use Descriptive Names**
```python
# ✅ Good
lora_path = paths.get_lora_path("alice_ppo_final_v2")

# ❌ Bad
lora_path = paths.get_lora_path("model1")
```

### 3. **Organize by Experiment**
```python
# Create experiment-specific directories
exp_path = paths.get_experiment_path("ppo_training_v1")
exp_checkpoint_path = exp_path / "checkpoints"
exp_log_path = exp_path / "logs"
```

### 4. **Version Your Outputs**
```python
# Include timestamps or version numbers
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_path = paths.get_checkpoint_path(f"ppo_v1_{timestamp}")
```

## 🚨 Important Notes

### **Symbolic Links**
- `shared_models` and `shared_data` are symbolic links
- They point to the actual directories outside your project
- Don't delete these links - they're how you access shared resources

### **Path Resolution**
- All paths are resolved relative to your project root
- The path manager handles the full path resolution
- Use the path manager methods instead of string concatenation

### **Backup**
- Before running the migration, backup your project
- The migration script is safe but it's always good to have a backup

## 🔍 Troubleshooting

### **"Module not found" errors**
Make sure you're running scripts from the project root directory.

### **Permission errors with symbolic links**
On some systems, you might need to create the links manually or adjust permissions.

### **Paths not resolving correctly**
Check that the `src/paths.py` file is in the correct location relative to your project root.

## 📚 Additional Resources

- `migrate_structure.py` - Migration script
- `config_improved.yaml` - Example configuration
- `src/paths.py` - Path management implementation
- `MIGRATION_SUMMARY.md` - Summary of what was moved

---

**Need help?** Check the migration summary or run the migration script with `--help` for more options.
