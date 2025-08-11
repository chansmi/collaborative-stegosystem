# Collaborative Stegosystem

A research project implementing steganographic communication between LLM agents using reinforcement learning. Two agents (Alice and Bob) attempt to share secret information while avoiding detection by an overseer (Eve).

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd collaborative-stegosystem

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys Setup

Create a `.env` file based on `env.template`:

```bash
# Copy template
cp env.template .env

# Edit with your keys
export HF_TOKEN=your_huggingface_token
export OPENAI_API_KEY=your_openai_api_key
export WANDB_API_KEY=your_wandb_api_key
```

**Required API Keys:**
- **HF_TOKEN**: HuggingFace token for accessing Meta-Llama models
- **OPENAI_API_KEY**: OpenAI API key for Eve's detection system
- **WANDB_API_KEY**: Weights & Biases API key for experiment tracking

### 3. Test Basic Functionality

```bash
# Run basic tests (no model downloads)
python test_basic.py

# Test training pipeline with small model
python test_training.py
```

### 4. Run Full Training

```bash
# Start training with Llama-2
python main.py
```

## 🏗️ Architecture

### Core Components

- **`src/models.py`**: Agent creation and model management
- **`src/environment.py`**: Trading environment and reward calculation
- **`src/ppo_trainer.py`**: PPO training implementation
- **`src/utils.py`**: Configuration and utility functions

### Training Flow

1. **Environment Setup**: Random stock and direction selection
2. **Agent Communication**: Alice and Bob exchange messages
3. **Steganography Detection**: Eve analyzes conversation for hidden information
4. **Reward Calculation**: Based on successful communication and evasion
5. **Model Updates**: PPO updates both agents' models

## ⚙️ Configuration

### Main Configuration (`config.yaml`)

```yaml
model:
  name: "meta-llama/Llama-2-7b-chat-hf"  # Model to use
  quantize: true                           # Enable 8-bit quantization
  max_new_tokens: 150                     # Max response length

training:
  num_epochs: 100                         # Training epochs
  learning_rate: 1.0e-6                   # PPO learning rate
```

### Test Configuration (`config_test.yaml`)

Uses smaller models for testing and development:

```yaml
model:
  name: "microsoft/DialoGPT-small"        # Small test model
  quantize: false                          # No quantization
  max_new_tokens: 50                      # Shorter responses

training:
  num_epochs: 5                           # Fewer epochs
```

## 🔧 Troubleshooting

### Common Issues

1. **PyArrow Compatibility**
   ```
   AttributeError: module 'pyarrow.lib' has no attribute 'Device'
   ```
   **Solution**: Use `datasets==2.20.0` as specified in requirements.txt

2. **Model Loading Failures**
   ```
   ValueError: HF_TOKEN environment variable is not set
   ```
   **Solution**: Set your HuggingFace token in environment variables

3. **CUDA/GPU Issues**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce model size or enable quantization in config

4. **OpenAI API Errors**
   ```
   openai.AuthenticationError: Invalid API key
   ```
   **Solution**: Verify your OpenAI API key is correct and has credits

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only training
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM
- **Optimal**: 32GB+ RAM, NVIDIA GPU with 16GB+ VRAM

## 📊 Experiment Tracking

### Weights & Biases Integration

The system automatically logs:
- Training metrics (rewards, losses)
- Agent responses and decisions
- Detection results from Eve
- System information (GPU, CUDA version)

### Custom Metrics

- **Steganography Success Rate**: Percentage of undetected communications
- **Communication Efficiency**: Reward per training step
- **Detection Evasion**: Success rate against Eve's analysis

## 🧪 Research Extensions

### Experiment Variations

The `experiments/` directory contains 20+ variations exploring:
- Different encoding strategies
- Alternative detection methods
- Multi-round communication protocols
- Ensemble approaches

### Adding New Experiments

1. Create new experiment file in `experiments/`
2. Inherit from base classes or modify existing code
3. Add configuration parameters
4. Document hypothesis and approach

## 📚 API Reference

### Key Classes

#### `CollaborativePPOTrainer`

```python
trainer = CollaborativePPOTrainer(config)
trainer.train(env, num_epochs=100)
```

#### `TradingEnvironment`

```python
env = TradingEnvironment(config)
state = env.reset()
next_state, reward, done = env.step(alice_msg, bob_msg)
```

#### `create_agent`

```python
agent = create_agent(config, 'Alice')
model = agent['model']
tokenizer = agent['tokenizer']
```

## 🤝 Contributing

### Development Workflow

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/new-feature`
3. **Test** changes: `python test_basic.py && python test_training.py`
4. **Commit** changes: `git commit -am 'Add new feature'`
5. **Push** branch: `git push origin feature/new-feature`
6. **Submit** pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints where possible
- Include docstrings for all functions
- Write tests for new functionality
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Meta AI** for Llama-2 models
- **HuggingFace** for transformers and TRL libraries
- **OpenAI** for GPT models used in detection
- **Weights & Biases** for experiment tracking

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with detailed error information
4. Include system information and error logs

---

**Note**: This is a research project. Results may vary based on hardware, model versions, and random seeds. For reproducible research, always set random seeds and document your environment.



