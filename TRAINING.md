# Training Guide

This guide explains how to train the collaborative stegosystem using different approaches.

## 🚀 **Quick Start Training**

### 1. **Basic PPO Training** (Recommended for beginners)
```bash
# Quick test (3 epochs)
python experiments/ppo_approach.py --quick

# Standard training (10 epochs)
python experiments/ppo_approach.py --epochs 10

# Full training (100 epochs)
python experiments/ppo_approach.py --full
```

### 2. **Using Different Configurations**
```bash
# Use GPU configuration
python experiments/ppo_approach.py --config config_gpu.yaml

# Use test configuration
python experiments/ppo_approach.py --config config_test.yaml

# Use main configuration
python experiments/ppo_approach.py --config config.yaml
```

## 🎯 **Training Approaches**

### **PPO Approach** (Default)
- **File**: `experiments/ppo_approach.py`
- **Description**: Standard PPO implementation using TRL library
- **Best for**: Most use cases, production training
- **Command**: `python experiments/ppo_approach.py`

### **Custom Approaches**
- **Encoder-Decoder**: `python experiments/encoder_decoder_approach.py`
- **Custom PPO**: `python experiments/custom_ppo_approach.py`
- **GPT Overseer**: `python experiments/gpt_overseer_approach.py`

## ⚙️ **Configuration Options**

### **Model Settings**
```yaml
model:
  name: "meta-llama/Llama-2-7b-chat-hf"  # Model to use
  quantize: true                           # Enable 8-bit quantization
  max_new_tokens: 150                     # Max response length
```

### **Training Settings**
```yaml
training:
  num_epochs: 100                         # Number of training epochs
  learning_rate: 1.0e-6                   # Learning rate
  batch_size: 1                           # Batch size
```

### **Environment Settings**
```yaml
env:
  max_rounds: 1                           # Rounds per episode
  stocks: ['AAPL', 'GOOGL', 'MSFT']      # Available stocks
  reward:
    correct: 1.0                          # Reward for correct communication
    partial: 0.5                          # Partial reward
    incorrect: -0.5                       # Penalty for incorrect
```

## 🔧 **Hardware Requirements**

### **CPU Training** (Development/Testing)
- **RAM**: 16GB+
- **Storage**: 50GB+
- **Use**: `config_test.yaml` with small models

### **GPU Training** (Production)
- **VRAM**: 16GB+ (24GB+ recommended)
- **RAM**: 32GB+
- **Storage**: 100GB+ SSD
- **Use**: `config_gpu.yaml` with Llama-2

## 📊 **Monitoring Training**

### **Weights & Biases Integration**
```bash
# Training metrics are automatically logged
# Check your W&B dashboard for:
# - Training rewards
# - Agent responses
# - Detection results
# - System metrics
```

### **Local Logging**
```bash
# Check the results directory
ls results/

# View training logs
cat results/training_log.txt

# Check saved models
ls models/
```

## 🚨 **Common Issues & Solutions**

### **Out of Memory**
```bash
# Enable quantization
config['model']['quantize'] = True

# Reduce batch size
config['ppo']['batch_size'] = 1

# Use smaller model
config['model']['name'] = "microsoft/DialoGPT-small"
```

### **Slow Training**
```bash
# Check GPU utilization
nvidia-smi

# Increase batch size if memory allows
config['ppo']['batch_size'] = 4

# Enable mixed precision
# (automatically handled by transformers)
```

### **Model Loading Failures**
```bash
# Verify API keys
echo $HF_TOKEN
echo $WANDB_API_KEY
echo $OPENAI_API_KEY

# Check internet connection
# Ensure sufficient disk space
```

## 📈 **Training Progress**

### **What to Expect**
1. **Epoch 1-10**: Agents learning basic communication
2. **Epoch 10-50**: Improving steganography techniques
3. **Epoch 50-100**: Fine-tuning and optimization

### **Success Metrics**
- **Steganography Success Rate**: >80%
- **Detection Evasion Rate**: >70%
- **Communication Accuracy**: >90%

### **When to Stop**
- **Early Stopping**: If success rate plateaus
- **Overfitting**: If validation performance decreases
- **Convergence**: When rewards stabilize

## 🔄 **Advanced Training**

### **Hyperparameter Tuning**
```bash
# Test different learning rates
python experiments/ppo_approach.py --epochs 20 --config config_lr_test.yaml

# Test different batch sizes
python experiments/ppo_approach.py --epochs 20 --config config_batch_test.yaml
```

### **Multi-GPU Training**
```bash
# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1

# Use distributed training config
python experiments/ppo_approach.py --config config_multi_gpu.yaml
```

### **Transfer Learning**
```bash
# Load pre-trained models
config['model']['load_pretrained'] = "models/previous_training"

# Continue training
python experiments/ppo_approach.py --config config_transfer.yaml
```

## 📝 **Training Checklist**

Before starting training:
- [ ] API keys are set (`HF_TOKEN`, `WANDB_API_KEY`, `OPENAI_API_KEY`)
- [ ] Dependencies are installed (`pip install -r requirements.txt`)
- [ ] Configuration file is correct
- [ ] Sufficient disk space available
- [ ] GPU drivers installed (for GPU training)

During training:
- [ ] Monitor GPU memory usage
- [ ] Check training metrics in W&B
- [ ] Save models periodically
- [ ] Monitor for overfitting

After training:
- [ ] Evaluate model performance
- [ ] Save final models
- [ ] Document training parameters
- [ ] Analyze results

## 🎯 **Next Steps**

1. **Start with quick test**: `python experiments/ppo_approach.py --quick`
2. **Run standard training**: `python experiments/ppo_approach.py --epochs 10`
3. **Deploy to GPU**: Use `config_gpu.yaml` on GPU infrastructure
4. **Experiment with approaches**: Try different experimental methods
5. **Scale up**: Increase epochs and model size for production

## 📁 **File Management**

### **What Gets Created During Training**
- **Models**: Saved to `models/` directory (not tracked in git)
- **Results**: Training logs saved to `results/` directory (not tracked in git)
- **WandB**: Experiment tracking data (not tracked in git)
- **Cache**: HuggingFace model cache (not tracked in git)

### **Git Best Practices**
- ✅ **Track**: Source code, configs, documentation
- ❌ **Don't Track**: Models, results, logs, cache files
- 🔄 **Update**: .gitignore ensures large files stay out of version control

This keeps your repository clean and focused on code rather than large binary files.

---

**Need Help?** Check the troubleshooting section or create an issue in the repository.
