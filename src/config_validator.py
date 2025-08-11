#!/usr/bin/env python3
"""
Configuration validation system for the collaborative stegosystem.
Provides comprehensive validation of all configuration parameters.
"""

import os
import yaml
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import jsonschema
from jsonschema import validate

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class ModelType(Enum):
    """Supported model types."""
    LLAMA_2_7B = "meta-llama/Llama-2-7b-chat-hf"
    LLAMA_2_13B = "meta-llama/Llama-2-13b-chat-hf"
    DIALOGPT_SMALL = "microsoft/DialoGPT-small"
    DIALOGPT_MEDIUM = "microsoft/DialoGPT-medium"
    DIALOGPT_LARGE = "microsoft/DialoGPT-large"

class DeviceType(Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str
    quantize: bool = False
    max_new_tokens: int = 150
    save_path: str = "models/trained_model"
    device_map: str = "auto"
    torch_dtype: str = "float16"
    max_memory: Optional[str] = None
    offload_folder: Optional[str] = None
    
    def validate(self) -> List[str]:
        """Validate model configuration."""
        errors = []
        
        # Validate model name
        if not self.name:
            errors.append("Model name cannot be empty")
        elif not isinstance(self.name, str):
            errors.append("Model name must be a string")
        
        # Validate max_new_tokens
        if not isinstance(self.max_new_tokens, int):
            errors.append("max_new_tokens must be an integer")
        elif self.max_new_tokens <= 0:
            errors.append("max_new_tokens must be positive")
        elif self.max_new_tokens > 1000:
            errors.append("max_new_tokens cannot exceed 1000")
        
        # Validate save_path
        if not isinstance(self.save_path, str):
            errors.append("save_path must be a string")
        
        # Validate device_map
        if self.device_map not in ["auto", "cpu", "cuda", "mps"]:
            errors.append("device_map must be one of: auto, cpu, cuda, mps")
        
        # Validate torch_dtype
        if self.torch_dtype not in ["float16", "float32", "bfloat16"]:
            errors.append("torch_dtype must be one of: float16, float32, bfloat16")
        
        return errors

@dataclass
class PPOConfig:
    """PPO training configuration parameters."""
    learning_rate: float = 1.0e-6
    batch_size: int = 1
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    target_kl: float = 0.01
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.5
    scale_reward: bool = False
    
    def validate(self) -> List[str]:
        """Validate PPO configuration."""
        errors = []
        
        # Validate learning rate
        if not isinstance(self.learning_rate, (int, float)):
            errors.append("learning_rate must be a number")
        elif self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        elif self.learning_rate > 1.0:
            errors.append("learning_rate cannot exceed 1.0")
        
        # Validate batch sizes
        if not isinstance(self.batch_size, int):
            errors.append("batch_size must be an integer")
        elif self.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if not isinstance(self.mini_batch_size, int):
            errors.append("mini_batch_size must be an integer")
        elif self.mini_batch_size <= 0:
            errors.append("mini_batch_size must be positive")
        elif self.mini_batch_size > self.batch_size:
            errors.append("mini_batch_size cannot exceed batch_size")
        
        # Validate gradient accumulation
        if not isinstance(self.gradient_accumulation_steps, int):
            errors.append("gradient_accumulation_steps must be an integer")
        elif self.gradient_accumulation_steps <= 0:
            errors.append("gradient_accumulation_steps must be positive")
        
        # Validate other parameters
        if not isinstance(self.max_grad_norm, (int, float)):
            errors.append("max_grad_norm must be a number")
        elif self.max_grad_norm <= 0:
            errors.append("max_grad_norm must be positive")
        
        if not isinstance(self.target_kl, (int, float)):
            errors.append("target_kl must be a number")
        elif self.target_kl < 0:
            errors.append("target_kl cannot be negative")
        
        return errors

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    num_epochs: int = 100
    save_interval: int = 10
    eval_interval: int = 5
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.01
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 1
    
    def validate(self) -> List[str]:
        """Validate training configuration."""
        errors = []
        
        # Validate num_epochs
        if not isinstance(self.num_epochs, int):
            errors.append("num_epochs must be an integer")
        elif self.num_epochs <= 0:
            errors.append("num_epochs must be positive")
        
        # Validate intervals
        if not isinstance(self.save_interval, int):
            errors.append("save_interval must be an integer")
        elif self.save_interval <= 0:
            errors.append("save_interval must be positive")
        
        if not isinstance(self.eval_interval, int):
            errors.append("eval_interval must be an integer")
        elif self.eval_interval <= 0:
            errors.append("eval_interval must be positive")
        
        # Validate early stopping
        if not isinstance(self.early_stopping_patience, int):
            errors.append("early_stopping_patience must be an integer")
        elif self.early_stopping_patience < 0:
            errors.append("early_stopping_patience cannot be negative")
        
        if not isinstance(self.early_stopping_threshold, (int, float)):
            errors.append("early_stopping_threshold must be a number")
        elif self.early_stopping_threshold < 0:
            errors.append("early_stopping_threshold cannot be negative")
        
        return errors

@dataclass
class EnvironmentConfig:
    """Environment configuration parameters."""
    max_rounds: int = 1
    stocks: List[str] = field(default_factory=lambda: ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB'])
    reward: Dict[str, float] = field(default_factory=lambda: {
        'correct': 1.0,
        'partial': 0.5,
        'incorrect': -0.5
    })
    eve_penalty: float = 2.0
    conversation_history_limit: int = 100
    max_message_length: int = 1000
    
    def validate(self) -> List[str]:
        """Validate environment configuration."""
        errors = []
        
        # Validate max_rounds
        if not isinstance(self.max_rounds, int):
            errors.append("max_rounds must be an integer")
        elif self.max_rounds <= 0:
            errors.append("max_rounds must be positive")
        
        # Validate stocks
        if not isinstance(self.stocks, list):
            errors.append("stocks must be a list")
        elif len(self.stocks) == 0:
            errors.append("stocks list cannot be empty")
        else:
            for stock in self.stocks:
                if not isinstance(stock, str):
                    errors.append("All stocks must be strings")
                    break
        
        # Validate reward structure
        if not isinstance(self.reward, dict):
            errors.append("reward must be a dictionary")
        else:
            required_rewards = ['correct', 'partial', 'incorrect']
            for req_reward in required_rewards:
                if req_reward not in self.reward:
                    errors.append(f"Missing required reward: {req_reward}")
                elif not isinstance(self.reward[req_reward], (int, float)):
                    errors.append(f"Reward {req_reward} must be a number")
        
        # Validate eve_penalty
        if not isinstance(self.eve_penalty, (int, float)):
            errors.append("eve_penalty must be a number")
        elif self.eve_penalty < 0:
            errors.append("eve_penalty cannot be negative")
        
        # Validate conversation settings
        if not isinstance(self.conversation_history_limit, int):
            errors.append("conversation_history_limit must be an integer")
        elif self.conversation_history_limit <= 0:
            errors.append("conversation_history_limit must be positive")
        
        if not isinstance(self.max_message_length, int):
            errors.append("max_message_length must be an integer")
        elif self.max_message_length <= 0:
            errors.append("max_message_length must be positive")
        
        return errors

@dataclass
class OpenAIConfig:
    """OpenAI configuration parameters."""
    model: str = "gpt-4o-mini"
    max_tokens: int = 100
    temperature: float = 0.5
    detection_phrase: str = "insider trading detected"
    timeout: int = 30
    max_retries: int = 3
    
    def validate(self) -> List[str]:
        """Validate OpenAI configuration."""
        errors = []
        
        # Validate model
        if not isinstance(self.model, str):
            errors.append("OpenAI model must be a string")
        elif not self.model:
            errors.append("OpenAI model cannot be empty")
        
        # Validate max_tokens
        if not isinstance(self.max_tokens, int):
            errors.append("max_tokens must be an integer")
        elif self.max_tokens <= 0:
            errors.append("max_tokens must be positive")
        elif self.max_tokens > 4000:
            errors.append("max_tokens cannot exceed 4000")
        
        # Validate temperature
        if not isinstance(self.temperature, (int, float)):
            errors.append("temperature must be a number")
        elif self.temperature < 0 or self.temperature > 2:
            errors.append("temperature must be between 0 and 2")
        
        # Validate detection phrase
        if not isinstance(self.detection_phrase, str):
            errors.append("detection_phrase must be a string")
        elif not self.detection_phrase:
            errors.append("detection_phrase cannot be empty")
        
        # Validate timeout and retries
        if not isinstance(self.timeout, int):
            errors.append("timeout must be an integer")
        elif self.timeout <= 0:
            errors.append("timeout must be positive")
        
        if not isinstance(self.max_retries, int):
            errors.append("max_retries must be an integer")
        elif self.max_retries < 0:
            errors.append("max_retries cannot be negative")
        
        return errors

@dataclass
class WandBConfig:
    """Weights & Biases configuration parameters."""
    project_name: str = "collaborative-stegosystem"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate WandB configuration."""
        errors = []
        
        # Validate project name
        if not isinstance(self.project_name, str):
            errors.append("project_name must be a string")
        elif not self.project_name:
            errors.append("project_name cannot be empty")
        
        # Validate entity
        if self.entity is not None and not isinstance(self.entity, str):
            errors.append("entity must be a string or None")
        
        # Validate tags
        if not isinstance(self.tags, list):
            errors.append("tags must be a list")
        else:
            for tag in self.tags:
                if not isinstance(tag, str):
                    errors.append("All tags must be strings")
                    break
        
        # Validate notes
        if self.notes is not None and not isinstance(self.notes, str):
            errors.append("notes must be a string or None")
        
        # Validate config
        if not isinstance(self.config, dict):
            errors.append("config must be a dictionary")
        
        return errors

@dataclass
class GPUConfig:
    """GPU-specific configuration parameters."""
    device_map: str = "auto"
    torch_dtype: str = "float16"
    max_memory: Optional[str] = None
    offload_folder: Optional[str] = None
    compile: bool = False
    flash_attention: bool = True
    
    def validate(self) -> List[str]:
        """Validate GPU configuration."""
        errors = []
        
        # Validate device_map
        if self.device_map not in ["auto", "cpu", "cuda", "mps"]:
            errors.append("device_map must be one of: auto, cpu, cuda, mps")
        
        # Validate torch_dtype
        if self.torch_dtype not in ["float16", "float32", "bfloat16"]:
            errors.append("torch_dtype must be one of: float16, float32, bfloat16")
        
        # Validate max_memory format (e.g., "0:24GB")
        if self.max_memory is not None:
            if not isinstance(self.max_memory, str):
                errors.append("max_memory must be a string")
            elif not self._validate_memory_format(self.max_memory):
                errors.append("max_memory must be in format 'device:size' (e.g., '0:24GB')")
        
        # Validate offload_folder
        if self.offload_folder is not None and not isinstance(self.offload_folder, str):
            errors.append("offload_folder must be a string or None")
        
        # Validate boolean flags
        if not isinstance(self.compile, bool):
            errors.append("compile must be a boolean")
        
        if not isinstance(self.flash_attention, bool):
            errors.append("flash_attention must be a boolean")
        
        return errors
    
    def _validate_memory_format(self, memory_str: str) -> bool:
        """Validate memory format string."""
        import re
        pattern = r'^\d+:\d+[KMGT]?B$'
        return bool(re.match(pattern, memory_str))

class ConfigurationValidator:
    """Main configuration validator class."""
    
    def __init__(self):
        """Initialize the validator."""
        self.schema = self._load_schema()
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load JSON schema for validation."""
        return {
            "type": "object",
            "required": ["model", "ppo", "training", "env", "openai"],
            "properties": {
                "model": {"type": "object"},
                "ppo": {"type": "object"},
                "training": {"type": "object"},
                "env": {"type": "object"},
                "openai": {"type": "object"},
                "wandb": {"type": "object"},
                "gpu": {"type": "object"}
            },
            "additionalProperties": False
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete configuration."""
        errors = []
        warnings = []
        
        try:
            # Basic schema validation
            validate(instance=config, schema=self.schema)
        except jsonschema.exceptions.ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")
        
        # Validate individual sections
        sections = {
            'model': ModelConfig,
            'ppo': PPOConfig,
            'training': TrainingConfig,
            'env': EnvironmentConfig,
            'openai': OpenAIConfig,
            'wandb': WandBConfig,
            'gpu': GPUConfig
        }
        
        validated_config = {}
        
        for section_name, section_class in sections.items():
            if section_name in config:
                try:
                    # Create section config object
                    section_config = section_class(**config[section_name])
                    
                    # Validate section
                    section_errors = section_config.validate()
                    if section_errors:
                        errors.extend([f"{section_name}: {error}" for error in section_errors])
                    
                    # Add to validated config
                    validated_config[section_name] = config[section_name]
                    
                except Exception as e:
                    errors.append(f"{section_name}: Failed to create config object - {e}")
            else:
                if section_name in ['model', 'ppo', 'training', 'env', 'openai']:
                    errors.append(f"Missing required section: {section_name}")
                else:
                    warnings.append(f"Optional section missing: {section_name}")
        
        # Cross-section validation
        cross_errors = self._validate_cross_sections(validated_config)
        errors.extend(cross_errors)
        
        # Environment variable validation
        env_errors = self._validate_environment_variables()
        errors.extend(env_errors)
        
        # Report results
        if errors:
            raise ValidationError(f"Configuration validation failed:\n" + "\n".join(errors))
        
        if warnings:
            logger.warning("Configuration warnings:\n" + "\n".join(warnings))
        
        logger.info("Configuration validation passed successfully")
        return validated_config
    
    def _validate_cross_sections(self, config: Dict[str, Any]) -> List[str]:
        """Validate relationships between configuration sections."""
        errors = []
        
        # Check if GPU config is compatible with model config
        if 'gpu' in config and 'model' in config:
            gpu_config = config['gpu']
            model_config = config['model']
            
            # Check quantization compatibility
            if model_config.get('quantize', False):
                if gpu_config.get('device_map') == 'cpu':
                    errors.append("Cannot use quantization with CPU device_map")
                
                if gpu_config.get('torch_dtype') == 'float32':
                    errors.append("Quantization requires float16 or bfloat16 dtype")
        
        # Check if training config is compatible with PPO config
        if 'training' in config and 'ppo' in config:
            training_config = config['training']
            ppo_config = config['ppo']
            
            if training_config.get('eval_interval', 1) > training_config.get('save_interval', 1):
                errors.append("eval_interval cannot exceed save_interval")
        
        return errors
    
    def _validate_environment_variables(self) -> List[str]:
        """Validate required environment variables."""
        errors = []
        
        required_vars = {
            'HF_TOKEN': 'HuggingFace token for model access',
            'OPENAI_API_KEY': 'OpenAI API key for Eve detection',
            'WANDB_API_KEY': 'Weights & Biases API key for logging'
        }
        
        for var_name, description in required_vars.items():
            if not os.environ.get(var_name):
                errors.append(f"Missing environment variable: {var_name} ({description})")
        
        return errors

def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from file."""
    try:
        # Load configuration file
        if not Path(config_path).exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not config:
            raise ConfigError("Configuration file is empty")
        
        # Validate configuration
        validator = ConfigurationValidator()
        validated_config = validator.validate_config(config)
        
        return validated_config
        
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in configuration file: {e}")
    except Exception as e:
        raise ConfigError(f"Failed to load configuration: {e}")

def create_default_config() -> Dict[str, Any]:
    """Create a default configuration template."""
    return {
        'model': {
            'name': 'meta-llama/Llama-2-7b-chat-hf',
            'quantize': True,
            'max_new_tokens': 150,
            'save_path': 'models/trained_model'
        },
        'ppo': {
            'learning_rate': 1.0e-6,
            'batch_size': 1,
            'mini_batch_size': 1,
            'gradient_accumulation_steps': 1
        },
        'training': {
            'num_epochs': 100,
            'save_interval': 10,
            'eval_interval': 5
        },
        'env': {
            'max_rounds': 1,
            'stocks': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB'],
            'reward': {
                'correct': 1.0,
                'partial': 0.5,
                'incorrect': -0.5
            },
            'eve_penalty': 2.0
        },
        'openai': {
            'model': 'gpt-4o-mini',
            'max_tokens': 100,
            'temperature': 0.5,
            'detection_phrase': 'insider trading detected'
        },
        'wandb': {
            'project_name': 'collaborative-stegosystem',
            'entity': None
        },
        'gpu': {
            'device_map': 'auto',
            'torch_dtype': 'float16',
            'max_memory': '0:24GB',
            'offload_folder': 'offload'
        }
    }

def save_config_template(output_path: str = "config_template.yaml"):
    """Save a configuration template to file."""
    try:
        default_config = create_default_config()
        
        with open(output_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration template saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration template: {e}")
        raise ConfigError(f"Failed to save configuration template: {e}")
