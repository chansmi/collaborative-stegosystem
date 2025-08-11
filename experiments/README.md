# Steganography Experiment Approaches

This directory contains different experimental approaches for the collaborative stegosystem. Each approach explores a different method of implementing steganographic communication between agents.

## 🎯 **Core Training Methods**

### 1. **PPO (Proximal Policy Optimization)**
- **File**: `ppo_approach.py`
- **Description**: Standard PPO implementation using TRL library
- **Use Case**: Baseline training method for agent coordination

### 2. **Encoder-Decoder Approach**
- **File**: `encoder_decoder_approach.py`
- **Description**: Separate encoder and decoder models for steganography
- **Use Case**: When you need explicit encoding/decoding control

### 3. **Custom PPO Implementation**
- **File**: `custom_ppo_approach.py`
- **Description**: Custom PPO implementation without TRL dependency
- **Use Case**: When you need full control over PPO implementation

## 🔍 **Detection Methods**

### 1. **GPT Overseer**
- **File**: `gpt_overseer_approach.py`
- **Description**: Uses GPT models for insider trading detection
- **Use Case**: Advanced detection using large language models

### 2. **Rule-Based Detection**
- **File**: `rule_based_detection.py`
- **Description**: Simple rule-based detection system
- **Use Case**: Fast, lightweight detection for development

## 📊 **Training Enhancements**

### 1. **Improved Learning**
- **File**: `improved_learning_approach.py`
- **Description**: Enhanced learning algorithms and techniques
- **Use Case**: When standard training isn't sufficient

### 2. **Segmented Training**
- **File**: `segmented_training_approach.py`
- **Description**: Breaks training into manageable segments
- **Use Case**: For very long training sessions or large models

### 3. **Validation Approaches**
- **File**: `validation_approaches.py`
- **Description**: Different validation strategies during training
- **Use Case**: Ensuring training quality and preventing overfitting

## 🎬 **Domain-Specific Approaches**

### 1. **Movie Review Domain**
- **File**: `movie_review_domain.py`
- **Description**: Steganography in movie review context
- **Use Case**: Testing in specific conversational domains

### 2. **Financial Domain**
- **File**: `financial_domain.py`
- **Description**: Steganography in financial communication
- **Use Case**: Real-world financial applications

## 🚀 **Getting Started**

1. **Choose your approach** based on your research goals
2. **Review the implementation** in the corresponding file
3. **Modify parameters** in the configuration files
4. **Run experiments** using the main training script

## 📝 **Adding New Approaches**

1. **Create a new file** with descriptive name
2. **Document the approach** in this README
3. **Follow the existing code structure** for consistency
4. **Test thoroughly** before adding to experiments

---

**Note**: All experiments use the same core infrastructure but explore different training and detection strategies.
