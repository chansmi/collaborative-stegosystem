# Deployment Guide for Distributed Computing

This guide covers deploying the collaborative stegosystem to GPU-enabled distributed computing infrastructure.

## 🚀 Quick Deployment

### 1. Environment Setup

```bash
# Clone and setup
git clone <repository-url>
cd collaborative-stegosystem

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN=your_huggingface_token_here
export WANDB_API_KEY=your_wandb_api_key_here
export OPENAI_API_KEY=your_openai_api_key_here
```

### 2. GPU Configuration

Use the GPU-optimized configuration:

```bash
python main.py --config config_gpu.yaml
```

## 🖥️ Hardware Requirements

### Minimum GPU Setup
- **VRAM**: 16GB+ (for Llama-2-7b with quantization)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ SSD storage

### Recommended GPU Setup
- **VRAM**: 24GB+ (for optimal performance)
- **RAM**: 64GB+ system memory
- **Storage**: 500GB+ NVMe SSD

### GPU Options
- **NVIDIA RTX 4090**: 24GB VRAM, excellent for development
- **NVIDIA RTX 3090**: 24GB VRAM, good value
- **NVIDIA A100**: 40GB/80GB VRAM, production deployment
- **NVIDIA H100**: 80GB VRAM, high-performance training

## ☁️ Cloud Deployment Options

### 1. Google Colab Pro+ (Development)
```python
# Install dependencies
!pip install -r requirements.txt

# Set environment variables
import os
os.environ['HF_TOKEN'] = 'your_huggingface_token_here'
os.environ['WANDB_API_KEY'] = 'your_wandb_api_key_here'
os.environ['OPENAI_API_KEY'] = 'your_openai_api_key_here'

# Run training
!python main.py --config config_gpu.yaml
```

### 2. AWS EC2 (Production)
```bash
# Launch p3.2xlarge or p3.8xlarge instance
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type p3.2xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxx

# Connect and setup
ssh -i your-key.pem ubuntu@your-instance-ip
sudo apt update && sudo apt install -y python3-pip
pip3 install -r requirements.txt
```

### 3. Google Cloud Platform
```bash
# Create instance with GPU
gcloud compute instances create stego-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --image-family=debian-11-gpu \
  --image-project=debian-cloud

# Install CUDA and dependencies
gcloud compute ssh stego-gpu --zone=us-central1-a
sudo apt update && sudo apt install -y nvidia-cuda-toolkit
pip3 install -r requirements.txt
```

### 4. Azure ML
```python
from azureml.core import Workspace, Environment, Experiment
from azureml.core.compute import ComputeTarget

# Create environment
env = Environment.from_conda_specification(
    name='stego-env',
    file_path='environment.yml'
)

# Submit training job
experiment = Experiment(workspace, 'stego-training')
run = experiment.submit(
    ScriptRunConfig(
        source_directory='.',
        script='main.py',
        arguments=['--config', 'config_gpu.yaml'],
        compute_target=compute_target,
        environment=env
    )
)
```

## 🐳 Docker Deployment

### 1. Create Dockerfile
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy source code
COPY . .

# Set environment variables
ENV HF_TOKEN=your_huggingface_token_here
ENV WANDB_API_KEY=your_wandb_api_key_here
ENV OPENAI_API_KEY=your_openai_api_key_here

# Run training
CMD ["python3", "main.py", "--config", "config_gpu.yaml"]
```

### 2. Build and Run
```bash
# Build image
docker build -t stego-gpu .

# Run with GPU access
docker run --gpus all -it stego-gpu
```

## 🔧 Performance Optimization

### 1. Memory Management
```yaml
# In config_gpu.yaml
model:
  quantize: true  # 8-bit quantization
  max_memory: "0:24GB"  # Limit GPU memory
  offload_folder: "offload"  # CPU offloading if needed
```

### 2. Batch Size Optimization
```yaml
ppo:
  batch_size: 4  # Increase for better GPU utilization
  mini_batch_size: 2
  gradient_accumulation_steps: 2
```

### 3. Mixed Precision Training
```python
# Enable in trainer
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
with autocast():
    # Training code
```

## 📊 Monitoring and Logging

### 1. Weights & Biases Integration
```python
# Automatic logging in main.py
wandb.init(
    project=config['wandb']['project_name'],
    entity=config['wandb']['entity'],
    config=config
)

# Custom metrics
wandb.log({
    'steganography_success_rate': success_rate,
    'detection_evasion_rate': evasion_rate,
    'gpu_memory_usage': torch.cuda.memory_allocated() / 1e9
})
```

### 2. System Monitoring
```bash
# GPU monitoring
nvidia-smi -l 1

# Memory monitoring
watch -n 1 'free -h && df -h'

# Process monitoring
htop
```

## 🚨 Troubleshooting

### Common GPU Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or enable quantization
   # Check nvidia-smi for memory usage
   ```

2. **Model Offloading Errors**
   ```bash
   # Ensure sufficient GPU memory
   # Use quantization: config['model']['quantize'] = True
   ```

3. **Flash Attention Issues**
   ```bash
   # Install flash-attn or disable in config
   # pip install flash-attn --no-build-isolation
   ```

### Performance Issues

1. **Slow Training**
   - Check GPU utilization with `nvidia-smi`
   - Verify batch size and learning rate
   - Enable mixed precision training

2. **Memory Issues**
   - Reduce model size or use quantization
   - Enable gradient checkpointing
   - Use CPU offloading for large models

## 🔄 Continuous Deployment

### 1. GitHub Actions
```yaml
name: Deploy to GPU
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to GPU instance
        run: |
          # Deployment commands
```

### 2. Automated Scaling
```python
# Auto-scaling based on queue length
import boto3

def scale_instances(queue_length):
    if queue_length > 10:
        # Launch new GPU instance
        ec2 = boto3.client('ec2')
        ec2.run_instances(
            ImageId='ami-0c02fb55956c7d316',
            InstanceType='p3.2xlarge',
            MinCount=1,
            MaxCount=1
        )
```

## 📈 Scaling Strategies

### 1. Horizontal Scaling
- Multiple GPU instances processing different experiments
- Load balancing across instances
- Shared model storage (S3, EFS)

### 2. Vertical Scaling
- Larger GPU instances (A100, H100)
- Increased batch sizes
- Model parallelism for very large models

### 3. Hybrid Approach
- Development on smaller instances
- Production training on large instances
- Auto-scaling based on demand

## 🎯 Next Steps

1. **Test on GPU infrastructure** using the provided configurations
2. **Optimize hyperparameters** for your specific hardware
3. **Set up monitoring** and alerting for production deployment
4. **Implement auto-scaling** for cost optimization
5. **Add experiment tracking** for research reproducibility

---

**Note**: This system is designed for research and development. For production deployment, consider additional security measures, monitoring, and backup strategies.
