# models.py
import re
import os
import logging
import time
import hashlib
from typing import Dict, Any, Optional, Union
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead
import torch
import json

# Configure logging
logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Custom exception for model-related errors."""
    pass

class ModelCache:
    """Manages model caching to prevent repeated downloads."""
    
    def __init__(self, cache_dir: str = "~/.cache/huggingface"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_file = self.cache_dir / "model_cache_index.json"
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict[str, Any]:
        """Load cache index from file."""
        try:
            if self.cache_index_file.exists():
                with open(self.cache_index_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
        return {}
    
    def _save_cache_index(self):
        """Save cache index to file."""
        try:
            with open(self.cache_index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def get_cache_key(self, model_name: str, config_hash: str) -> str:
        """Generate cache key for model configuration."""
        return f"{model_name}_{config_hash}"
    
    def is_cached(self, cache_key: str) -> bool:
        """Check if model is cached."""
        return cache_key in self.cache_index
    
    def add_to_cache(self, cache_key: str, model_info: Dict[str, Any]):
        """Add model to cache index."""
        self.cache_index[cache_key] = {
            **model_info,
            'cached_at': time.time()
        }
        self._save_cache_index()
        logger.info(f"Model {cache_key} added to cache")
    
    def get_cache_info(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached model information."""
        return self.cache_index.get(cache_key)

# Global model cache instance
model_cache = ModelCache()

def create_agent(config: Dict[str, Any], role: str) -> Dict[str, Any]:
    """Create an agent with comprehensive error handling and caching."""
    
    try:
        # Validate inputs
        if not config or not isinstance(config, dict):
            raise ModelError("Invalid configuration: must be a dictionary")
        
        if not role or not isinstance(role, str):
            raise ModelError(f"Invalid role: {role}")
        
        # Check authentication
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ModelError("HF_TOKEN environment variable is not set. Please set it to access Meta-Llama models.")
        
        model_name = config.get('model', {}).get('name')
        if not model_name:
            raise ModelError("Model name not specified in configuration")
        
        logger.info(f"Creating {role} agent with model: {model_name}")
        
        # Generate configuration hash for caching
        config_hash = _generate_config_hash(config)
        cache_key = model_cache.get_cache_key(model_name, config_hash)
        
        # Check if model is already cached
        if model_cache.is_cached(cache_key):
            logger.info(f"Using cached model for {role}: {cache_key}")
            return _load_cached_model(cache_key, role)
        
        # Create new model
        model_info = _create_model_from_scratch(config, role, cache_key)
        
        # Add to cache
        model_cache.add_to_cache(cache_key, model_info)
        
        return model_info
        
    except Exception as e:
        logger.error(f"Failed to create {role} agent: {e}")
        raise ModelError(f"Agent creation failed for {role}: {e}")

def _generate_config_hash(config: Dict[str, Any]) -> str:
    """Generate hash for configuration to enable caching."""
    try:
        # Extract relevant config parts for hashing
        config_str = json.dumps({
            'model_name': config.get('model', {}).get('name'),
            'quantize': config.get('model', {}).get('quantize', False),
            'max_new_tokens': config.get('model', {}).get('max_new_tokens', 50),
            'ppo_config': config.get('ppo', {}),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }, sort_keys=True)
        
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
        
    except Exception as e:
        logger.warning(f"Failed to generate config hash: {e}")
        return "default"

def _create_model_from_scratch(config: Dict[str, Any], role: str, cache_key: str) -> Dict[str, Any]:
    """Create model from scratch with comprehensive error handling."""
    
    model_name = config['model']['name']
    quantization_config = _setup_quantization(config)
    device_map = _setup_device_mapping(config)
    
    # Load base model with fallbacks
    model = _load_base_model(model_name, quantization_config, device_map, config)
    
    # Convert to PPO model
    model = _convert_to_ppo_model(model, role)
    
    # Load and configure tokenizer
    tokenizer = _load_tokenizer(model_name, model, config)
    
    # Validate model and tokenizer
    _validate_model_components(model, tokenizer, role)
    
    model_info = {
        'model': model,
        'tokenizer': tokenizer,
        'role': role,
        'model_name': model_name,
        'cache_key': cache_key,
        'created_at': time.time()
    }
    
    logger.info(f"Successfully created {role} agent with model {model_name}")
    return model_info

def _setup_quantization(config: Dict[str, Any]) -> Optional[BitsAndBytesConfig]:
    """Setup quantization configuration with hardware detection."""
    
    if not config['model'].get('quantize', False):
        return None
    
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if 'NVIDIA' in gpu_name:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                logger.info(f"Using 8-bit quantization for NVIDIA GPU: {gpu_name}")
                return quantization_config
            else:
                logger.info(f"Using dynamic quantization for GPU: {gpu_name}")
                return None
        else:
            logger.info("Using dynamic quantization for CPU")
            return None
            
    except Exception as e:
        logger.warning(f"Quantization setup failed: {e}, falling back to no quantization")
        return None

def _setup_device_mapping(config: Dict[str, Any]) -> str:
    """Setup device mapping strategy."""
    
    if torch.cuda.is_available():
        # Check GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU memory available: {gpu_memory:.2f} GB")
        
        if gpu_memory >= 24:
            return "auto"  # Let transformers handle placement
        elif gpu_memory >= 16:
            return "auto"  # Conservative placement
        else:
            return "auto"  # Minimal placement
    else:
        return "cpu"

def _load_base_model(model_name: str, quantization_config: Optional[BitsAndBytesConfig], 
                    device_map: str, config: Dict[str, Any]) -> AutoModelForCausalLM:
    """Load base model with comprehensive fallback strategy."""
    
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading base model (attempt {attempt + 1}/{max_retries})")
            
            # Try with flash attention 2 first
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if quantization_config is None else None,
                    attn_implementation="flash_attention_2",
                    token=os.environ.get("HF_TOKEN"),
                )
                logger.info("Successfully loaded model with flash_attention_2")
                return model
                
            except Exception as e:
                logger.info(f"Flash attention 2 failed: {e}, trying without it")
                
                # Try without flash attention
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if quantization_config is None else None,
                    token=os.environ.get("HF_TOKEN"),
                )
                logger.info("Successfully loaded model without flash_attention_2")
                return model
                
        except Exception as e:
            last_error = e
            logger.warning(f"Model loading attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
    
    # All attempts failed
    raise ModelError(f"Failed to load model {model_name} after {max_retries} attempts: {last_error}")

def _convert_to_ppo_model(model: AutoModelForCausalLM, role: str) -> AutoModelForCausalLMWithValueHead:
    """Convert base model to PPO format with error handling."""
    
    try:
        logger.info(f"Converting {role} model to PPO format...")
        
        ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        
        # Configure PPO model
        ppo_model.config.use_cache = False
        ppo_model.gradient_checkpointing_enable()
        
        logger.info(f"Successfully converted {role} model to PPO format")
        return ppo_model
        
    except Exception as e:
        logger.error(f"Failed to convert to PPO model: {e}")
        
        # Try alternative conversion method
        try:
            logger.info("Trying alternative PPO conversion method...")
            
            # Save and reload as PPO model
            temp_path = f"temp_{role}_model"
            model.save_pretrained(temp_path)
            
            ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(temp_path)
            ppo_model.config.use_cache = False
            ppo_model.gradient_checkpointing_enable()
            
            # Clean up temp files
            import shutil
            shutil.rmtree(temp_path, ignore_errors=True)
            
            logger.info(f"Successfully converted {role} model using alternative method")
            return ppo_model
            
        except Exception as e2:
            logger.error(f"Alternative PPO conversion also failed: {e2}")
            raise ModelError(f"PPO conversion failed for {role}: {e}")

def _load_tokenizer(model_name: str, model: Union[AutoModelForCausalLM, AutoModelForCausalLMWithValueHead], 
                   config: Dict[str, Any]) -> AutoTokenizer:
    """Load and configure tokenizer with validation."""
    
    try:
        logger.info(f"Loading tokenizer for {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            token=os.environ.get("HF_TOKEN")
        )
        
        # Configure tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if not hasattr(tokenizer, 'model_max_length') or tokenizer.model_max_length is None:
            tokenizer.model_max_length = 512
        
        # Ensure tokenizer is on the same device as model
        if hasattr(model, 'device'):
            tokenizer.device = model.device
        
        logger.info(f"Successfully loaded and configured tokenizer")
        return tokenizer
        
    except Exception as e:
        logger.error(f"Tokenizer loading failed: {e}")
        raise ModelError(f"Tokenizer loading failed: {e}")

def _validate_model_components(model: Union[AutoModelForCausalLM, AutoModelForCausalLMWithValueHead], 
                             tokenizer: AutoTokenizer, role: str):
    """Validate model and tokenizer components."""
    
    try:
        # Check model
        if not hasattr(model, 'generate'):
            raise ModelError(f"Model for {role} missing generate method")
        
        if not hasattr(model, 'config'):
            raise ModelError(f"Model for {role} missing config")
        
        # Check tokenizer
        if not hasattr(tokenizer, 'encode'):
            raise ModelError(f"Tokenizer for {role} missing encode method")
        
        if not hasattr(tokenizer, 'decode'):
            raise ModelError(f"Tokenizer for {role} missing decode method")
        
        # Test basic functionality
        test_input = "Hello, world!"
        test_tokens = tokenizer.encode(test_input, return_tensors="pt")
        test_output = tokenizer.decode(test_tokens[0])
        
        if not test_output:
            raise ModelError(f"Tokenizer for {role} failed basic encode/decode test")
        
        logger.info(f"Model components validation passed for {role}")
        
    except Exception as e:
        logger.error(f"Model components validation failed for {role}: {e}")
        raise ModelError(f"Validation failed for {role}: {e}")

def _load_cached_model(cache_key: str, role: str) -> Dict[str, Any]:
    """Load cached model (placeholder for future implementation)."""
    # This would implement loading from cache
    # For now, we'll create a new model
    logger.info(f"Cache loading not yet implemented, creating new model for {role}")
    raise ModelError("Cache loading not yet implemented")

def extract_decision(response: str) -> str:
    """Extract decision from agent response with enhanced parsing."""
    
    if not response or not isinstance(response, str):
        return "unknown"
    
    try:
        # Normalize response
        response_lower = response.lower().strip()
        
        # Look for stock mentions
        stock_patterns = [
            r'\b(aapl|apple)\b',
            r'\b(googl|google)\b', 
            r'\b(msft|microsoft)\b',
            r'\b(amzn|amazon)\b',
            r'\b(fb|facebook|meta)\b',
            r'\b(tsla|tesla)\b',
            r'\b(nvda|nvidia)\b'
        ]
        
        for pattern in stock_patterns:
            if re.search(pattern, response_lower):
                return re.search(pattern, response_lower).group(1)
        
        # Look for direction mentions
        direction_patterns = [
            r'\b(up|rise|gain|positive|bullish)\b',
            r'\b(down|fall|drop|negative|bearish)\b'
        ]
        
        for pattern in direction_patterns:
            if re.search(pattern, response_lower):
                return re.search(pattern, response_lower).group(1)
        
        # Look for market sentiment
        if any(word in response_lower for word in ['bull', 'bullish', 'optimistic', 'positive']):
            return "up"
        elif any(word in response_lower for word in ['bear', 'bearish', 'pessimistic', 'negative']):
            return "down"
        
        # Default fallback
        return "neutral"
        
    except Exception as e:
        logger.warning(f"Failed to extract decision from response: {e}")
        return "unknown"

def get_model_info(model: Union[AutoModelForCausalLM, AutoModelForCausalLMWithValueHead]) -> Dict[str, Any]:
    """Get comprehensive information about a model."""
    
    try:
        info = {
            'model_type': type(model).__name__,
            'device': str(next(model.parameters()).device) if hasattr(model, 'parameters') else 'unknown',
            'dtype': str(next(model.parameters()).dtype) if hasattr(model, 'parameters') else 'unknown',
            'num_parameters': sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') else 0,
            'config': {}
        }
        
        if hasattr(model, 'config'):
            config = model.config
            info['config'] = {
                'vocab_size': getattr(config, 'vocab_size', 'unknown'),
                'hidden_size': getattr(config, 'hidden_size', 'unknown'),
                'num_attention_heads': getattr(config, 'num_attention_heads', 'unknown'),
                'num_hidden_layers': getattr(config, 'num_hidden_layers', 'unknown'),
                'max_position_embeddings': getattr(config, 'max_position_embeddings', 'unknown')
            }
        
        return info
        
    except Exception as e:
        logger.warning(f"Failed to get model info: {e}")
        return {'error': str(e)}

def cleanup_model_resources(model: Union[AutoModelForCausalLM, AutoModelForCausalLMWithValueHead]):
    """Clean up model resources to free memory."""
    
    try:
        if hasattr(model, 'cpu'):
            model.cpu()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model resources cleaned up successfully")
        
    except Exception as e:
        logger.warning(f"Failed to cleanup model resources: {e}")