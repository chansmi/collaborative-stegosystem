#!/usr/bin/env python3
"""
Quick Test Script for Collaborative Stegosystem
Runs minimal tests to verify basic functionality.
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test if basic modules can be imported"""
    print("🔍 Testing basic imports...")
    
    try:
        from src.models import create_agents
        from src.environment import TradingEnvironment
        from src.ppo_trainer import CollaborativePPOTrainer
        from src.utils import load_config
        print("✅ All source modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("🔍 Testing configuration loading...")
    
    try:
        from src.utils import load_config
        config = load_config('config_test_enhanced.yaml')
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_environment_creation():
    """Test environment creation"""
    print("🔍 Testing environment creation...")
    
    try:
        from src.environment import TradingEnvironment
        from src.utils import load_config
        
        config = load_config('config_test_enhanced.yaml')
        env = TradingEnvironment(config)
        print("✅ Environment created successfully")
        return True
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("🔍 Testing model creation...")
    
    try:
        from src.models import create_agents
        from src.utils import load_config
        
        config = load_config('config_test_enhanced.yaml')
        alice, bob = create_agents(config)
        print("✅ Models created successfully")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_openai_connection():
    """Test OpenAI API connection"""
    print("🔍 Testing OpenAI connection...")
    
    try:
        import openai
        
        # Load API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  OPENAI_API_KEY not found in environment")
            print("💡 Set it with: export OPENAI_API_KEY=your_openai_api_key_here")
            print("💡 Or create a .env file from env.template")
            return False
        
        client = openai.OpenAI(api_key=api_key)
        
        # Simple test
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=5
        )
        
        print(f"✅ OpenAI connection successful: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False

def main():
    """Run all quick tests"""
    print("🚀 Running Quick Tests for Collaborative Stegosystem")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration Loading", test_config_loading),
        ("Environment Creation", test_environment_creation),
        ("Model Creation", test_model_creation),
        ("OpenAI Connection", test_openai_connection)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 QUICK TEST RESULTS")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All quick tests passed! Basic functionality verified.")
        print("You can now run the full test deployment with:")
        print("python run_test_deployment.py")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
