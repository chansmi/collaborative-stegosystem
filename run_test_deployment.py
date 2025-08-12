#!/usr/bin/env python3
"""
Test Deployment Runner for Collaborative Stegosystem
This script sets up and runs a test deployment without launching the full training.
"""

import os
import sys
import yaml
import logging
import torch
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def setup_logging(config):
    """Set up logging configuration"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logs directory if it doesn't exist
    log_file = log_config.get('file', 'logs/test_run.log')
    Path(log_file).parent.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) if log_config.get('console', True) else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_environment():
    """Load environment variables"""
    # Try to load from .env file first
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded environment from {env_file}")
    else:
        print("⚠️  .env file not found, using system environment variables")
        print("💡 Create a .env file from env.template for easier configuration")
    
    # Check required environment variables
    required_vars = ['OPENAI_API_KEY', 'HF_TOKEN', 'WANDB_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("💡 Set them with:")
        print("   export OPENAI_API_KEY=your_openai_api_key_here")
        print("   export HF_TOKEN=your_huggingface_token_here")
        print("   export WANDB_API_KEY=your_wandb_api_key_here")
        print("💡 Or create a .env file from env.template")
        return False
    
    print("✅ Environment variables loaded successfully")
    return True

def validate_config(config):
    """Validate the configuration file"""
    logger = logging.getLogger(__name__)
    
    required_sections = ['wandb', 'env', 'model', 'ppo', 'training', 'openai']
    missing_sections = [section for section in required_sections if section not in config]
    
    if missing_sections:
        logger.error(f"Missing required configuration sections: {missing_sections}")
        return False
    
    # Validate model configuration
    if not config['model'].get('name'):
        logger.error("Model name not specified in configuration")
        return False
    
    # Validate training configuration
    if config['training'].get('num_epochs', 0) <= 0:
        logger.error("Number of epochs must be greater than 0")
        return False
    
    logger.info("✅ Configuration validation passed")
    return True

def check_dependencies():
    """Check if all required dependencies are available"""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        import transformers
        import trl
        import peft
        import openai
        import wandb
        
        logger.info(f"✅ PyTorch version: {torch.__version__}")
        logger.info(f"✅ Transformers version: {transformers.__version__}")
        logger.info(f"✅ TRL version: {trl.__version__}")
        logger.info(f"✅ PEFT version: {peft.__version__}")
        logger.info(f"✅ OpenAI version: {openai.__version__}")
        logger.info(f"✅ WandB version: {wandb.__version__}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

def check_hardware():
    """Check hardware availability"""
    logger = logging.getLogger(__name__)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"✅ CUDA available: {device_count} GPU(s)")
        logger.info(f"✅ Primary GPU: {device_name}")
        
        # Check GPU memory
        for i in range(device_count):
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"✅ GPU {i} memory: {memory:.1f} GB")
    else:
        logger.warning("⚠️  CUDA not available, will use CPU")
    
    # Check system memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"✅ System RAM: {memory.total / 1024**3:.1f} GB")
        logger.info(f"✅ Available RAM: {memory.available / 1024**3:.1f} GB")
    except ImportError:
        logger.warning("⚠️  psutil not available, cannot check system memory")
    
    return True

def setup_directories():
    """Create necessary directories"""
    logger = logging.getLogger(__name__)
    
    directories = [
        'models',
        'logs',
        'results',
        'data',
        'checkpoints'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")
    
    return True

def test_model_loading(config):
    """Test if the specified model can be loaded"""
    logger = logging.getLogger(__name__)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = config['model']['name']
        logger.info(f"🔄 Testing model loading: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"✅ Tokenizer loaded successfully")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for testing
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"✅ Model loaded successfully")
        
        # Test forward pass
        test_input = tokenizer("Hello, how are you?", return_tensors="pt")
        if torch.cuda.is_available():
            test_input = {k: v.cuda() for k, v in test_input.items()}
        
        with torch.no_grad():
            output = model(**test_input)
        
        logger.info(f"✅ Model forward pass successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

def test_openai_connection(config):
    """Test OpenAI API connection"""
    logger = logging.getLogger(__name__)
    
    try:
        import openai
        
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        logger.info("🔄 Testing OpenAI API connection...")
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model=config['openai']['model'],
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        logger.info(f"✅ OpenAI API connection successful")
        logger.info(f"✅ Response received: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        logger.error(f"❌ OpenAI API connection failed: {e}")
        return False

def main():
    """Main test deployment function"""
    print("🚀 Starting Collaborative Stegosystem Test Deployment")
    print("=" * 60)
    
    # Load configuration
    config_file = "config_test_enhanced.yaml"
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded configuration from {config_file}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False
    
    # Setup logging
    logger = setup_logging(config)
    
    # Run all checks
    checks = [
        ("Environment Variables", load_environment),
        ("Configuration Validation", lambda: validate_config(config)),
        ("Dependencies", check_dependencies),
        ("Hardware", check_hardware),
        ("Directory Setup", setup_directories),
        ("Model Loading", lambda: test_model_loading(config)),
        ("OpenAI Connection", lambda: test_openai_connection(config))
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n🔍 Running {check_name} check...")
        try:
            result = check_func()
            results.append((check_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {check_name}")
        except Exception as e:
            results.append((check_name, False))
            print(f"❌ FAILED: {check_name} - {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST DEPLOYMENT SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {check_name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Test deployment is ready.")
        print("You can now run the actual training with:")
        print("python experiments/ppo_approach.py --config config_test_enhanced.yaml")
        return True
    else:
        print(f"\n⚠️  {total - passed} check(s) failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
