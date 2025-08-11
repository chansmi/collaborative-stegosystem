import torch
import wandb
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from trl import PPOTrainer, PPOConfig
from src.models import create_agent, extract_decision
from transformers import LogitsProcessorList, LogitsProcessor
import random
import time
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class InfNanLogitsProcessor(LogitsProcessor):
    """Processes logits to handle infinite and NaN values."""
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            logger.warning("Detected NaN or Inf in logits, replacing with -1e8")
            scores = torch.where(
                torch.isnan(scores) | torch.isinf(scores), 
                torch.full_like(scores, -1e8), 
                scores
            )
        return scores

def filter_non_finite(d: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out non-finite values from dictionary for logging."""
    return {k: v for k, v in d.items() if not isinstance(v, (float, int)) or np.isfinite(v)}

class TrainingError(Exception):
    """Custom exception for training-related errors."""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class CollaborativePPOTrainer:
    """Enhanced PPO trainer with comprehensive error handling and monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the trainer with enhanced error handling."""
        self.config = self._validate_config(config)
        self.device = self._setup_device()
        self.logger = logger
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
        # Initialize components with error handling
        self.alice = self._create_agent_safely('Alice')
        self.bob = self._create_agent_safely('Bob')
        
        # Create PPO trainers with error handling
        self.alice_trainer = self._create_ppo_trainer_safely('Alice')
        self.bob_trainer = self._create_ppo_trainer_safely('Bob')
        
        # Enhanced generation kwargs
        self.generation_kwargs = self._setup_generation_kwargs()
        
        # Training state tracking
        self.training_stats = {
            'epochs_completed': 0,
            'total_steps': 0,
            'best_reward': float('-inf'),
            'training_start_time': time.time(),
            'errors_encountered': 0
        }
        
        logger.info(f"Trainer initialized successfully on device: {self.device}")
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration parameters."""
        required_keys = ['model', 'ppo', 'training', 'env']
        
        for key in required_keys:
            if key not in config:
                raise ValidationError(f"Missing required config key: {key}")
        
        # Validate model config
        model_config = config['model']
        if 'name' not in model_config:
            raise ValidationError("Model name not specified in config")
        
        # Validate PPO config
        ppo_config = config['ppo']
        required_ppo_keys = ['learning_rate', 'batch_size', 'mini_batch_size']
        for key in required_ppo_keys:
            if key not in ppo_config:
                raise ValidationError(f"Missing PPO config key: {key}")
        
        # Validate training config
        training_config = config['training']
        if 'num_epochs' not in training_config:
            raise ValidationError("Number of epochs not specified in training config")
        
        # Validate environment config
        env_config = config['env']
        if 'stocks' not in env_config or 'reward' not in env_config:
            raise ValidationError("Invalid environment configuration")
        
        logger.info("Configuration validation passed")
        return config
    
    def _setup_device(self) -> torch.device:
        """Setup device with fallback strategy."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")
        
        return device
    
    def _set_random_seeds(self, seed: int = 42):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Random seeds set to {seed}")
    
    def _create_agent_safely(self, role: str) -> Dict[str, Any]:
        """Create agent with comprehensive error handling."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                agent = create_agent(self.config, role)
                logger.info(f"Successfully created {role} agent")
                return agent
            except Exception as e:
                logger.error(f"Failed to create {role} agent (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise TrainingError(f"Failed to create {role} agent after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _create_ppo_trainer_safely(self, role: str) -> PPOTrainer:
        """Create PPO trainer with error handling."""
        try:
            agent = getattr(self, role.lower())
            ppo_config = PPOConfig(**self.config['ppo'])
            
            trainer = PPOTrainer(
                config=ppo_config,
                model=agent['model'],
                tokenizer=agent['tokenizer']
            )
            
            logger.info(f"Successfully created PPO trainer for {role}")
            return trainer
            
        except Exception as e:
            logger.error(f"Failed to create PPO trainer for {role}: {e}")
            raise TrainingError(f"PPO trainer creation failed for {role}: {e}")
    
    def _setup_generation_kwargs(self) -> Dict[str, Any]:
        """Setup generation parameters with validation."""
        try:
            max_new_tokens = self.config['model'].get('max_new_tokens', 50)
            if max_new_tokens <= 0 or max_new_tokens > 1000:
                raise ValidationError(f"Invalid max_new_tokens: {max_new_tokens}")
            
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95,
                "temperature": 0.7,
                "pad_token_id": self.alice['tokenizer'].eos_token_id,
                "eos_token_id": self.alice['tokenizer'].eos_token_id,
                "logits_processor": LogitsProcessorList([InfNanLogitsProcessor()]),
                "repetition_penalty": 1.1,
                "length_penalty": 1.0
            }
            
            logger.info(f"Generation kwargs configured: max_tokens={max_new_tokens}")
            return kwargs
            
        except Exception as e:
            logger.error(f"Failed to setup generation kwargs: {e}")
            raise TrainingError(f"Generation kwargs setup failed: {e}")

    def generate_response(self, trainer: PPOTrainer, prompt: str) -> str:
        """Generate response with enhanced error handling and monitoring."""
        start_time = time.time()
        
        try:
            # Input validation
            if not prompt or not isinstance(prompt, str):
                raise ValidationError(f"Invalid prompt: {prompt}")
            
            # Tokenize input
            inputs = trainer.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            query_tensor = inputs.input_ids.squeeze(0)
            
            # Generate response with monitoring
            with torch.no_grad():
                response = trainer.generate(
                    query_tensor,
                    **self.generation_kwargs
                )
            
            # Decode response
            decoded_response = trainer.tokenizer.decode(
                response[0], 
                skip_special_tokens=True
            )
            
            # Clean response
            if decoded_response.startswith(prompt):
                decoded_response = decoded_response[len(prompt):].strip()
            
            if not decoded_response or decoded_response.isspace():
                decoded_response = "I understand the situation."
            
            # Log generation metrics
            generation_time = time.time() - start_time
            response_length = len(decoded_response.split())
            
            logger.debug(f"Generated response in {generation_time:.2f}s, length: {response_length}")
            
            return decoded_response
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Response generation failed after {generation_time:.2f}s: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return fallback response
            return "I need to think about this more carefully."
    
    def _log_training_metrics(self, epoch: int, step: int, reward: float, 
                             alice_decision: str, bob_decision: str, 
                             insider_trading_detected: bool):
        """Log comprehensive training metrics."""
        try:
            metrics = {
                'epoch': epoch,
                'step': step,
                'reward': reward,
                'alice_decision': alice_decision,
                'bob_decision': bob_decision,
                'insider_trading_detected': insider_trading_detected,
                'training_progress': step / self.config['training']['num_epochs'],
                'best_reward': self.training_stats['best_reward'],
                'errors_encountered': self.training_stats['errors_encountered']
            }
            
            # Filter out non-finite values for wandb
            safe_metrics = filter_non_finite(metrics)
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log(safe_metrics)
            
            # Log to file
            logger.info(f"Epoch {epoch}, Step {step}: Reward={reward:.3f}, "
                       f"Alice={alice_decision}, Bob={bob_decision}, "
                       f"Detected={insider_trading_detected}")
            
        except Exception as e:
            logger.error(f"Failed to log training metrics: {e}")
            self.training_stats['errors_encountered'] += 1

    def train_step(self, env, epoch: int) -> Tuple[float, Dict[str, Any]]:
        """Execute a single training step with enhanced error handling."""
        step_start_time = time.time()
        
        try:
            # Generate responses from both agents
            alice_prompt = env._get_state()
            alice_message = self.generate_response(self.alice_trainer, alice_prompt)
            
            bob_prompt = f"Alice said: {alice_message}\nWhat do you think?"
            bob_message = self.generate_response(self.bob_trainer, bob_prompt)
            
            # Execute environment step
            next_state, reward, done = env.step(alice_message, bob_message)
            
            # Extract decisions for logging
            alice_decision = extract_decision(alice_message)
            bob_decision = extract_decision(bob_message)
            
            # Update training statistics
            self.training_stats['total_steps'] += 1
            if reward > self.training_stats['best_reward']:
                self.training_stats['best_reward'] = reward
            
            # Log metrics
            insider_trading_detected = getattr(env, 'insider_trading_detected', False)
            self._log_training_metrics(epoch, self.training_stats['total_steps'], 
                                     reward, alice_decision, bob_decision, 
                                     insider_trading_detected)
            
            # Training step
            alice_loss = self.alice_trainer.step([alice_message], [reward])
            bob_loss = self.bob_trainer.step([bob_message], [reward])
            
            step_time = time.time() - step_start_time
            
            return reward, {
                'alice_loss': alice_loss,
                'bob_loss': bob_loss,
                'step_time': step_time,
                'alice_decision': alice_decision,
                'bob_decision': bob_decision
            }
            
        except Exception as e:
            step_time = time.time() - step_start_time
            logger.error(f"Training step failed after {step_time:.2f}s: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            self.training_stats['errors_encountered'] += 1
            
            # Return default values on error
            return 0.0, {
                'alice_loss': 0.0,
                'bob_loss': 0.0,
                'step_time': step_time,
                'error': str(e)
            }

    def _save_models(self, suffix: str = ""):
        """Save models with error handling and validation."""
        try:
            save_path = Path(self.config['model'].get('save_path', 'models'))
            save_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            alice_path = save_path / f"alice_model_{suffix}_{timestamp}"
            bob_path = save_path / f"bob_model_{suffix}_{timestamp}"
            
            # Save Alice model
            self.alice['model'].save_pretrained(alice_path)
            self.alice['tokenizer'].save_pretrained(alice_path)
            
            # Save Bob model
            self.bob['model'].save_pretrained(bob_path)
            self.bob['tokenizer'].save_pretrained(bob_path)
            
            logger.info(f"Models saved successfully: {alice_path}, {bob_path}")
            
            # Save training statistics
            stats_path = save_path / f"training_stats_{suffix}_{timestamp}.json"
            import json
            with open(stats_path, 'w') as f:
                json.dump(self.training_stats, f, indent=2)
            
            return str(alice_path), str(bob_path)
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            raise TrainingError(f"Model saving failed: {e}")

    def train(self, env, num_epochs: Optional[int] = None):
        """Execute training with comprehensive monitoring and error handling."""
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Target device: {self.device}")
        logger.info(f"Model: {self.config['model']['name']}")
        
        training_start_time = time.time()
        epoch_rewards = []
        
        try:
            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                epoch_reward = 0.0
                epoch_steps = 0
                
                logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
                
                # Reset environment
                env.reset()
                
                # Training loop for this epoch
                while not env.current_round >= env.max_rounds:
                    reward, step_info = self.train_step(env, epoch + 1)
                    epoch_reward += reward
                    epoch_steps += 1
                    
                    # Check for early stopping conditions
                    if self._should_stop_early(epoch_reward, epoch_steps):
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break
                
                # Epoch completion
                epoch_time = time.time() - epoch_start_time
                epoch_rewards.append(epoch_reward)
                
                logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s: "
                           f"Reward={epoch_reward:.3f}, Steps={epoch_steps}")
                
                # Periodic model saving
                if (epoch + 1) % 10 == 0:
                    self._save_models(f"epoch_{epoch + 1}")
                
                # Update training statistics
                self.training_stats['epochs_completed'] = epoch + 1
                
                # Log epoch summary
                if wandb.run is not None:
                    wandb.log({
                        'epoch': epoch + 1,
                        'epoch_reward': epoch_reward,
                        'epoch_steps': epoch_steps,
                        'epoch_time': epoch_time,
                        'cumulative_reward': sum(epoch_rewards),
                        'avg_reward': np.mean(epoch_rewards)
                    })
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise TrainingError(f"Training execution failed: {e}")
        finally:
            # Final model saving
            try:
                self._save_models("final")
                logger.info("Final models saved successfully")
            except Exception as e:
                logger.error(f"Failed to save final models: {e}")
            
            # Training summary
            total_time = time.time() - training_start_time
            logger.info(f"Training completed in {total_time:.2f}s")
            logger.info(f"Total epochs: {self.training_stats['epochs_completed']}")
            logger.info(f"Total steps: {self.training_stats['total_steps']}")
            logger.info(f"Best reward: {self.training_stats['best_reward']:.3f}")
            logger.info(f"Errors encountered: {self.training_stats['errors_encountered']}")
    
    def _should_stop_early(self, epoch_reward: float, epoch_steps: int) -> bool:
        """Determine if training should stop early."""
        # Stop if reward is consistently negative
        if epoch_reward < -10 and epoch_steps > 5:
            return True
        
        # Stop if too many errors
        if self.training_stats['errors_encountered'] > 50:
            return True
        
        return False