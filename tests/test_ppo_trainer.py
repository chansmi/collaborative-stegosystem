#!/usr/bin/env python3
"""
Unit tests for PPO Trainer module.
Tests error handling, validation, and core functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.ppo_trainer import (
    CollaborativePPOTrainer, 
    TrainingError, 
    ValidationError,
    InfNanLogitsProcessor,
    filter_non_finite
)

class TestInfNanLogitsProcessor(unittest.TestCase):
    """Test the InfNanLogitsProcessor class."""
    
    def test_normal_logits(self):
        """Test processing of normal logits."""
        processor = InfNanLogitsProcessor()
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.tensor([[0.1, 0.2, 0.3]])
        
        result = processor(input_ids, scores)
        self.assertTrue(torch.allclose(result, scores))
    
    def test_nan_logits(self):
        """Test processing of NaN logits."""
        processor = InfNanLogitsProcessor()
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.tensor([[0.1, float('nan'), 0.3]])
        
        result = processor(input_ids, scores)
        self.assertTrue(torch.isnan(result[0, 1]))
        self.assertEqual(result[0, 1].item(), -1e8)
    
    def test_inf_logits(self):
        """Test processing of infinite logits."""
        processor = InfNanLogitsProcessor()
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.tensor([[0.1, float('inf'), 0.3]])
        
        result = processor(input_ids, scores)
        self.assertEqual(result[0, 1].item(), -1e8)

class TestFilterNonFinite(unittest.TestCase):
    """Test the filter_non_finite function."""
    
    def test_normal_values(self):
        """Test filtering of normal values."""
        data = {'a': 1.0, 'b': 2.5, 'c': -3.0}
        result = filter_non_finite(data)
        self.assertEqual(result, data)
    
    def test_nan_values(self):
        """Test filtering of NaN values."""
        data = {'a': 1.0, 'b': float('nan'), 'c': 3.0}
        result = filter_non_finite(data)
        self.assertIn('a', result)
        self.assertIn('c', result)
        self.assertNotIn('b', result)
    
    def test_inf_values(self):
        """Test filtering of infinite values."""
        data = {'a': 1.0, 'b': float('inf'), 'c': 3.0}
        result = filter_non_finite(data)
        self.assertIn('a', result)
        self.assertIn('c', result)
        self.assertNotIn('b', result)
    
    def test_mixed_types(self):
        """Test filtering with mixed data types."""
        data = {'a': 1.0, 'b': 'string', 'c': [1, 2, 3], 'd': float('nan')}
        result = filter_non_finite(data)
        self.assertIn('a', result)
        self.assertIn('b', result)
        self.assertIn('c', result)
        self.assertNotIn('d', result)

class TestCollaborativePPOTrainer(unittest.TestCase):
    """Test the CollaborativePPOTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_config = {
            'model': {
                'name': 'test-model',
                'max_new_tokens': 100
            },
            'ppo': {
                'learning_rate': 1e-6,
                'batch_size': 1,
                'mini_batch_size': 1
            },
            'training': {
                'num_epochs': 10
            },
            'env': {
                'stocks': ['AAPL', 'GOOGL'],
                'reward': {'correct': 1.0, 'partial': 0.5, 'incorrect': -0.5}
            }
        }
        
        # Create temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.ppo_trainer.create_agent')
    @patch('src.ppo_trainer.PPOConfig')
    @patch('src.ppo_trainer.PPOTrainer')
    @patch('torch.cuda.is_available')
    def test_init_success(self, mock_cuda, mock_ppo_trainer, mock_ppo_config, mock_create_agent):
        """Test successful trainer initialization."""
        mock_cuda.return_value = False
        
        # Mock agent creation
        mock_alice = {'model': Mock(), 'tokenizer': Mock()}
        mock_bob = {'model': Mock(), 'tokenizer': Mock()}
        mock_create_agent.side_effect = [mock_alice, mock_bob]
        
        # Mock PPO components
        mock_ppo_config.return_value = Mock()
        mock_ppo_trainer.return_value = Mock()
        
        # Mock tokenizer attributes
        mock_alice['tokenizer'].eos_token_id = 1
        mock_bob['tokenizer'].eos_token_id = 1
        
        trainer = CollaborativePPOTrainer(self.valid_config)
        
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.device, torch.device('cpu'))
        self.assertEqual(trainer.training_stats['epochs_completed'], 0)
    
    def test_init_invalid_config(self):
        """Test initialization with invalid configuration."""
        invalid_config = {'model': {}}  # Missing required keys
        
        with self.assertRaises(ValidationError):
            CollaborativePPOTrainer(invalid_config)
    
    @patch('src.ppo_trainer.create_agent')
    @patch('torch.cuda.is_available')
    def test_init_agent_creation_failure(self, mock_cuda, mock_create_agent):
        """Test initialization when agent creation fails."""
        mock_cuda.return_value = False
        mock_create_agent.side_effect = Exception("Agent creation failed")
        
        with self.assertRaises(TrainingError):
            CollaborativePPOTrainer(self.valid_config)
    
    @patch('src.ppo_trainer.create_agent')
    @patch('src.ppo_trainer.PPOConfig')
    @patch('src.ppo_trainer.PPOTrainer')
    @patch('torch.cuda.is_available')
    def test_generate_response_success(self, mock_cuda, mock_ppo_trainer, mock_ppo_config, mock_create_agent):
        """Test successful response generation."""
        mock_cuda.return_value = False
        
        # Mock agent creation
        mock_alice = {'model': Mock(), 'tokenizer': Mock()}
        mock_bob = {'model': Mock(), 'tokenizer': Mock()}
        mock_create_agent.side_effect = [mock_alice, mock_bob]
        
        # Mock PPO components
        mock_ppo_config.return_value = Mock()
        mock_ppo_trainer.return_value = Mock()
        
        # Mock tokenizer attributes
        mock_alice['tokenizer'].eos_token_id = 1
        mock_bob['tokenizer'].eos_token_id = 1
        
        trainer = CollaborativePPOTrainer(self.valid_config)
        
        # Mock trainer.generate
        mock_trainer = Mock()
        mock_trainer.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_trainer.tokenizer = mock_alice['tokenizer']
        
        # Mock tokenizer methods
        mock_alice['tokenizer'].encode.return_value = Mock(input_ids=torch.tensor([[1, 2, 3]]))
        mock_alice['tokenizer'].decode.return_value = "Generated response"
        
        result = trainer.generate_response(mock_trainer, "Test prompt")
        
        self.assertEqual(result, "Generated response")
    
    def test_generate_response_invalid_input(self):
        """Test response generation with invalid input."""
        trainer = Mock()
        trainer.config = self.valid_config
        trainer.device = torch.device('cpu')
        trainer.alice = {'tokenizer': Mock(eos_token_id=1)}
        
        with self.assertRaises(ValidationError):
            trainer.generate_response(Mock(), None)
    
    @patch('src.ppo_trainer.create_agent')
    @patch('src.ppo_trainer.PPOConfig')
    @patch('src.ppo_trainer.PPOTrainer')
    @patch('torch.cuda.is_available')
    def test_train_step_success(self, mock_cuda, mock_ppo_trainer, mock_ppo_config, mock_create_agent):
        """Test successful training step."""
        mock_cuda.return_value = False
        
        # Mock agent creation
        mock_alice = {'model': Mock(), 'tokenizer': Mock()}
        mock_bob = {'model': Mock(), 'tokenizer': Mock()}
        mock_create_agent.side_effect = [mock_alice, mock_bob]
        
        # Mock PPO components
        mock_ppo_config.return_value = Mock()
        mock_ppo_trainer.return_value = Mock()
        
        # Mock tokenizer attributes
        mock_alice['tokenizer'].eos_token_id = 1
        mock_bob['tokenizer'].eos_token_id = 1
        
        trainer = CollaborativePPOTrainer(self.valid_config)
        
        # Mock environment
        mock_env = Mock()
        mock_env._get_state.return_value = {'target_stock': 'AAPL', 'target_direction': 'up'}
        mock_env.step.return_value = ({}, 1.0, False)
        mock_env.current_round = 0
        mock_env.max_rounds = 1
        mock_env.insider_trading_detected = False
        
        # Mock PPO trainers
        trainer.alice_trainer = Mock()
        trainer.bob_trainer = Mock()
        trainer.alice_trainer.step.return_value = 0.1
        trainer.bob_trainer.step.return_value = 0.2
        
        # Mock response generation
        trainer.generate_response = Mock(side_effect=["Alice response", "Bob response"])
        
        # Mock decision extraction
        with patch('src.ppo_trainer.extract_decision', side_effect=['AAPL', 'up']):
            reward, step_info = trainer.train_step(mock_env, 1)
        
        self.assertEqual(reward, 1.0)
        self.assertIn('alice_loss', step_info)
        self.assertIn('bob_loss', step_info)
    
    @patch('src.ppo_trainer.create_agent')
    @patch('src.ppo_trainer.PPOConfig')
    @patch('src.ppo_trainer.PPOTrainer')
    @patch('torch.cuda.is_available')
    def test_train_step_failure(self, mock_cuda, mock_ppo_trainer, mock_ppo_config, mock_create_agent):
        """Test training step with failure."""
        mock_cuda.return_value = False
        
        # Mock agent creation
        mock_alice = {'model': Mock(), 'tokenizer': Mock()}
        mock_bob = {'model': Mock(), 'tokenizer': Mock()}
        mock_create_agent.side_effect = [mock_alice, mock_bob]
        
        # Mock PPO components
        mock_ppo_config.return_value = Mock()
        mock_ppo_trainer.return_value = Mock()
        
        # Mock tokenizer attributes
        mock_alice['tokenizer'].eos_token_id = 1
        mock_bob['tokenizer'].eos_token_id = 1
        
        trainer = CollaborativePPOTrainer(self.valid_config)
        
        # Mock environment that raises exception
        mock_env = Mock()
        mock_env._get_state.side_effect = Exception("Environment error")
        
        reward, step_info = trainer.train_step(mock_env, 1)
        
        self.assertEqual(reward, 0.0)
        self.assertIn('error', step_info)
        self.assertEqual(trainer.training_stats['errors_encountered'], 1)
    
    @patch('src.ppo_trainer.create_agent')
    @patch('src.ppo_trainer.PPOConfig')
    @patch('src.ppo_trainer.PPOTrainer')
    @patch('torch.cuda.is_available')
    def test_should_stop_early(self, mock_cuda, mock_ppo_trainer, mock_ppo_config, mock_create_agent):
        """Test early stopping conditions."""
        mock_cuda.return_value = False
        
        # Mock agent creation
        mock_alice = {'model': Mock(), 'tokenizer': Mock()}
        mock_bob = {'model': Mock(), 'tokenizer': Mock()}
        mock_create_agent.side_effect = [mock_alice, mock_bob]
        
        # Mock PPO components
        mock_ppo_config.return_value = Mock()
        mock_ppo_trainer.return_value = Mock()
        
        # Mock tokenizer attributes
        mock_alice['tokenizer'].eos_token_id = 1
        mock_bob['tokenizer'].eos_token_id = 1
        
        trainer = CollaborativePPOTrainer(self.valid_config)
        
        # Test negative reward condition
        should_stop = trainer._should_stop_early(-15.0, 6)
        self.assertTrue(should_stop)
        
        # Test error threshold condition
        trainer.training_stats['errors_encountered'] = 55
        should_stop = trainer._should_stop_early(0.0, 1)
        self.assertTrue(should_stop)
        
        # Test normal condition
        trainer.training_stats['errors_encountered'] = 0
        should_stop = trainer._should_stop_early(1.0, 1)
        self.assertFalse(should_stop)

class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation functionality."""
    
    def test_valid_config(self):
        """Test validation of valid configuration."""
        valid_config = {
            'model': {'name': 'test-model'},
            'ppo': {
                'learning_rate': 1e-6,
                'batch_size': 1,
                'mini_batch_size': 1
            },
            'training': {'num_epochs': 10},
            'env': {
                'stocks': ['AAPL'],
                'reward': {'correct': 1.0, 'partial': 0.5, 'incorrect': -0.5}
            }
        }
        
        trainer = Mock()
        trainer._validate_config = CollaborativePPOTrainer._validate_config
        
        result = trainer._validate_config(valid_config)
        self.assertEqual(result, valid_config)
    
    def test_missing_model_config(self):
        """Test validation with missing model configuration."""
        invalid_config = {
            'ppo': {'learning_rate': 1e-6},
            'training': {'num_epochs': 10},
            'env': {'stocks': ['AAPL']}
        }
        
        trainer = Mock()
        trainer._validate_config = CollaborativePPOTrainer._validate_config
        
        with self.assertRaises(ValidationError):
            trainer._validate_config(invalid_config)
    
    def test_missing_ppo_config(self):
        """Test validation with missing PPO configuration."""
        invalid_config = {
            'model': {'name': 'test-model'},
            'training': {'num_epochs': 10},
            'env': {'stocks': ['AAPL']}
        }
        
        trainer = Mock()
        trainer._validate_config = CollaborativePPOTrainer._validate_config
        
        with self.assertRaises(ValidationError):
            trainer._validate_config(invalid_config)

if __name__ == '__main__':
    unittest.main()
