# Collaborative Stegosystem

A research project implementing steganographic communication between LLM agents using reinforcement learning. Two agents (Alice and Bob) attempt to share secret information while avoiding detection by an overseer (Eve).

## 🚀 **Quick Start Training**

### **1. Set Environment Variables**
```bash
export HF_TOKEN=your_huggingface_token_here
export WANDB_API_KEY=your_wandb_api_key_here
export OPENAI_API_KEY=your_openai_api_key_here
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run Training**
```bash
# Quick test (3 epochs)
python experiments/ppo_approach.py --quick

# Standard training (10 epochs)
python experiments/ppo_approach.py --epochs 10

# Full training (100 epochs)
python experiments/ppo_approach.py --full
```

## 🎯 **Training Methods (Obvious & Clear)**

### **PPO Approach** (Default - Use This First)
- **Command**: `python experiments/ppo_approach.py`
- **What it does**: Standard PPO training using TRL library
- **Best for**: Most use cases, production training
- **File**: `experiments/ppo_approach.py`
- **Notes**: Prompts avoid instructing covert behavior. Alice writes neutral market text; Bob outputs a discrete action line. Rewards are based on Bob’s action vs hidden state, with rule-based detection penalties for explicit leaks.

### **Other Approaches** (For Research)
- **Encoder-Decoder**: Separate encoding/decoding models
- **Custom PPO**: Full control over PPO implementation
- **GPT Overseer**: Advanced detection using GPT models

## 🏗️ **Project Structure**

```
collaborative-stegosystem/
├── src/                          # Core source code
│   ├── models.py                 # Agent creation & model management
│   ├── environment.py            # Trading environment & rewards
│   ├── ppo_trainer.py           # PPO training implementation
│   └── utils.py                 # Configuration & utilities
├── experiments/                  # Different training approaches
│   ├── ppo_approach.py          # 🎯 START HERE - Standard PPO
│   ├── README.md                # Approach descriptions
│   └── [other approaches]       # For research & experimentation
├── config.yaml                  # Main configuration
├── config_gpu.yaml              # GPU-optimized configuration
├── config_test.yaml             # Test configuration (small models)
├── TRAINING.md                  # 📚 Training guide
├── DEPLOYMENT.md                # 🚀 Deployment guide
└── requirements.txt             # Dependencies
```

## ⚙️ **Configuration Files**

### **Main Configuration** (`config.yaml`)
- Uses Llama-2-7b model
- Full training (100 epochs)
- Production settings

### **GPU Configuration** (`config_gpu.yaml`)
- Optimized for GPU training
- 8-bit quantization enabled
- Memory management settings

### **Test Configuration** (`config_test.yaml`)
- Small models for testing
- Fewer epochs
- CPU-friendly settings

## 🔧 **Hardware Requirements**

### **Development/Testing** (CPU)
- **RAM**: 16GB+
- **Storage**: 50GB+
- **Use**: `config_test.yaml`

### **Production Training** (GPU)
- **VRAM**: 16GB+ (24GB+ recommended)
- **RAM**: 32GB+
- **Storage**: 100GB+ SSD
- **Use**: `config_gpu.yaml`

## 📊 **What Gets Trained**

1. **Alice Agent**: Learns to communicate stock information secretly
2. **Bob Agent**: Learns to decode stock information secretly
3. **Steganography**: Both agents learn to hide information in natural conversation
4. **Detection Evasion**: Agents learn to avoid detection by Eve

## 🎓 **Training Process**

1. **Environment Setup**: Random stock and direction selection
2. **Agent Communication**: Alice and Bob exchange messages
3. **Steganography Detection**: Eve analyzes conversation for hidden information
4. **Reward Calculation**: Based on successful communication and evasion
5. **Model Updates**: PPO updates both agents' models

## 🚀 **Deployment Options**

### **Local Development**
```bash
python experiments/ppo_approach.py --quick
```

### **GPU Infrastructure**
```bash
# Single node with 4 GPUs (LLNL Lassen)
python experiments/ppo_approach.py --config config_gpu.yaml --epochs 20
```

### **Cloud Platforms**
- **Google Colab Pro+**: Free GPU access
- **AWS EC2**: p3.2xlarge or p3.8xlarge instances
- **Google Cloud**: GPU-enabled instances
- **Azure ML**: Managed GPU training

## 📚 **Documentation**

- **TRAINING.md**: Complete training guide
- **DEPLOYMENT.md**: GPU deployment instructions
- **experiments/README.md**: Approach descriptions

## 🔍 **Monitoring & Results**

### **Weights & Biases Integration**
- Training metrics automatically logged (set via `wandb` in config)
- Agent responses and decisions tracked
- Detection results from Eve
- System performance metrics

### **Local Results**
- Models saved to `models/` directory
- Training logs in `results/` directory
- Performance metrics and analysis

## 🚨 **Troubleshooting**

### **Common Issues**
1. **Out of Memory**: Enable quantization, reduce batch size, lower `max_new_tokens`
2. **Model Loading**: Verify API keys, check internet connection
3. **Slow Training**: Check GPU utilization, increase batch size if possible

### **Getting Help**
1. Check the troubleshooting sections in TRAINING.md
2. Verify environment variables are set
3. Check hardware requirements
4. Review error messages for specific issues

## 🎯 **Next Steps**

1. **Start Training**: `python experiments/ppo_approach.py --quick`
2. **Read TRAINING.md**: For detailed instructions
3. **Deploy to GPU**: Use DEPLOYMENT.md for GPU infrastructure
4. **Experiment**: Try different approaches in experiments/
5. **Scale Up**: Increase epochs and model size for production

## 📝 **Git and File Management**

### **What's Tracked in Git**
- Source code (`src/`)
- Configuration files (`*.yaml`)
- Documentation (`*.md`)
- Requirements (`requirements.txt`)
- Experiment approaches (`experiments/`)

### **What's NOT Tracked in Git** (Large Files)
- **Models**: `models/` directory (saved during training)
- **Results**: `results/` directory (training logs and outputs)
- **WandB**: `wandb/` directory (experiment tracking)
- **Cache**: `.cache/` and HuggingFace cache directories
- **Data**: Large data files and datasets

This keeps the repository lightweight and focused on code rather than large binary files.

---

**Note**: This is a research project. Results may vary based on hardware, model versions, and random seeds. For reproducible research, always set random seeds and document your environment.



