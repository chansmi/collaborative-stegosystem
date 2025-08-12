# Test Deployment Checklist

## 🚀 Pre-Deployment Setup

### 1. Environment Variables ✅
**Option A: Export directly (recommended for testing)**
```bash
export OPENAI_API_KEY=your_openai_api_key_here
export HF_TOKEN=your_huggingface_token_here
export WANDB_API_KEY=your_wandb_api_key_here
```

**Option B: Use .env file (recommended for development)**
```bash
# Copy template
cp env.template .env

# Edit .env file with your actual API keys
nano .env
```

**Required**: 
- OPENAI_API_KEY
- HF_TOKEN  
- WANDB_API_KEY

### 2. Dependencies ✅
- [ ] All packages from `requirements.txt` installed
- [ ] Python 3.10+ available
- [ ] PyTorch with CUDA support (if using GPU)

### 3. Configuration Files ✅
- [ ] `config_test_enhanced.yaml` created
- [ ] Test-specific settings configured
- [ ] WandB disabled for testing

## 🔍 Test Execution Order

### Step 1: Quick Tests
```bash
# Run basic functionality tests
python quick_test.py
```
**Expected**: All 5 tests should pass

### Step 2: Full Test Deployment
```bash
# Run comprehensive deployment tests
python run_test_deployment.py
```
**Expected**: All 7 checks should pass

### Step 3: Manual Verification
```bash
# Check if directories were created
ls -la models/ logs/ results/ data/ checkpoints/

# Verify environment variables
echo $OPENAI_API_KEY

# Check if .env file exists (optional)
ls -la .env
```

## 📋 What Each Test Checks

### Quick Tests (`quick_test.py`)
1. **Basic Imports**: Source modules can be imported
2. **Configuration Loading**: YAML config loads correctly
3. **Environment Creation**: Trading environment initializes
4. **Model Creation**: AI agents can be created
5. **OpenAI Connection**: API key works and connection successful

### Full Deployment Tests (`run_test_deployment.py`)
1. **Environment Variables**: API keys and settings loaded
2. **Configuration Validation**: All required sections present
3. **Dependencies**: All packages available with correct versions
4. **Hardware**: GPU/CPU detection and memory assessment
5. **Directory Setup**: Required folders created
6. **Model Loading**: HuggingFace model downloads and loads
7. **OpenAI Connection**: API communication works

## 🚨 Common Issues & Solutions

### Issue: OpenAI API Connection Failed
**Symptoms**: SSL certificate errors, connection timeouts
**Solutions**:
- Check if behind corporate proxy (LLNL systems)
- Verify API key format and validity
- Test with `curl -v https://api.openai.com`

### Issue: Model Loading Failed
**Symptoms**: HuggingFace download errors, memory issues
**Solutions**:
- Check internet connection
- Verify sufficient disk space
- Use smaller model for testing

### Issue: Import Errors
**Symptoms**: ModuleNotFoundError, ImportError
**Solutions**:
- Install missing packages: `pip install -r requirements.txt`
- Check Python path and virtual environment
- Verify file structure

## 🎯 Success Criteria

### All Tests Pass ✅
- Quick tests: 5/5 passed
- Deployment tests: 7/7 passed
- No critical errors in logs
- All directories created successfully

### Ready for Training ✅
- Environment fully configured
- Models can be loaded
- OpenAI API accessible
- Configuration validated
- Logging system working

## 🚀 Next Steps After Successful Test

1. **Run Quick Training Test**:
   ```bash
   python experiments/ppo_approach.py --config config_test_enhanced.yaml --quick
   ```

2. **Monitor Training**:
   - Check `logs/test_run.log`
   - Monitor GPU usage (if available)
   - Watch for any runtime errors

3. **Scale Up**:
   - Increase epochs in config
   - Enable WandB logging
   - Use larger models

## 📊 Expected Test Output

### Successful Quick Test
```
🚀 Running Quick Tests for Collaborative Stegosystem
==================================================
🔍 Testing basic imports...
✅ All source modules imported successfully
🔍 Testing configuration loading...
✅ Configuration loaded successfully
🔍 Testing environment creation...
✅ Environment created successfully
🔍 Testing model creation...
✅ Models created successfully
🔍 Testing OpenAI connection...
✅ OpenAI connection successful: Hello

==================================================
📊 QUICK TEST RESULTS
==================================================
✅ PASSED: Basic Imports
✅ PASSED: Configuration Loading
✅ PASSED: Environment Creation
✅ PASSED: Model Creation
✅ PASSED: OpenAI Connection

Overall: 5/5 tests passed

🎉 All quick tests passed! Basic functionality verified.
```

### Successful Deployment Test
```
🚀 Starting Collaborative Stegosystem Test Deployment
============================================================
✅ Loaded configuration from config_test_enhanced.yaml
✅ Environment variables loaded successfully
✅ Configuration validation passed
✅ Dependencies check passed
✅ Hardware check passed
✅ Directory setup completed
✅ Model loading test passed
✅ OpenAI connection test passed

============================================================
📊 TEST DEPLOYMENT SUMMARY
============================================================
✅ PASSED: Environment Variables
✅ PASSED: Configuration Validation
✅ PASSED: Dependencies
✅ PASSED: Hardware
✅ PASSED: Directory Setup
✅ PASSED: Model Loading
✅ PASSED: OpenAI Connection

Overall: 7/7 checks passed

🎉 All checks passed! Test deployment is ready.
```

## 🔧 Troubleshooting Commands

```bash
# Check Python environment
python --version
pip list | grep -E "(torch|transformers|trl|peft)"

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test OpenAI API directly
python -c "import openai; print(openai.__version__)"

# Check environment variables
echo $OPENAI_API_KEY
echo $HF_TOKEN
echo $WANDB_API_KEY
env | grep -E "(OPENAI|HF_TOKEN|WANDB)"

# Check disk space
df -h

# Check memory
free -h
```

## 📞 Getting Help

If tests continue to fail:
1. Check the error messages in detail
2. Verify all prerequisites are met
3. Check LLNL system status
4. Contact support if needed

**LLNL Support**:
- LC Support: (925) 422-4531
- LivIT: 925-424-4357
