#!/bin/bash
# Universal test submission script
# Automatically detects system and submits appropriate job

echo "🚀 Collaborative Stegosystem Test Submission"
echo "=========================================="

# Detect system type
if [[ "$(hostname)" == *"tioga"* ]]; then
    echo "📍 Detected Tioga system (LLNL)"
    echo "📋 Tioga uses Flux - you have two options:"
    echo ""
    echo "Option 1: Interactive allocation (recommended for testing)"
    echo "  flux alloc -n8 -N1 -t 30m"
    echo "  Then run: python run_test_deployment.py"
    echo ""
    echo "Option 2: Batch submission"
    echo "  flux batch submit_test_flux.sh"
    echo ""
    echo "Option 3: Direct execution (no attach needed)"
    echo "  flux run -n1 -N1 python run_test_deployment.py"
    echo ""
    echo "Which would you prefer?"
    read -p "Enter 1 for interactive, 2 for batch, 3 for direct: " choice
    
    if [ "$choice" = "1" ]; then
        echo "🚀 Starting interactive allocation..."
        echo "💡 After allocation, run: python run_test_deployment.py"
        flux alloc -n8 -N1 -t 30m
    elif [ "$choice" = "2" ]; then
        echo "📤 Submitting batch job..."
        chmod +x submit_test_flux.sh
        flux batch submit_test_flux.sh
    elif [ "$choice" = "3" ]; then
        echo "🚀 Running direct execution..."
        echo "💡 Setting environment variables and running test..."
        
        # Load environment variables from .env file
        if [ -f .env ]; then
            echo "✅ Loading environment from .env file"
            export $(cat .env | grep -v '^#' | xargs)
        else
            echo "❌ .env file not found!"
            echo "💡 Create .env file from env.template first:"
            echo "   cp env.template .env"
            echo "   nano .env  # Edit with your actual API keys"
            exit 1
        fi
        
        # Run the test directly
        flux run -n1 -N1 python run_test_deployment.py
    else
        echo "❌ Invalid choice. Exiting."
        exit 1
    fi
    
elif command -v sbatch >/dev/null 2>&1; then
    echo "📍 Detected Slurm system"
    echo "📋 Using Slurm script"
    
    # Make script executable
    chmod +x submit_test_slurm.sh
    
    # Submit with Slurm
    echo "📤 Submitting job to Slurm..."
    sbatch submit_test_slurm.sh
    
elif command -v flux >/dev/null 2>&1; then
    echo "📍 Detected Flux system"
    echo "📋 Using generic Flux script"
    
    # Make script executable
    chmod +x submit_test_flux.sh
    
    # Submit with Flux
    echo "📤 Submitting job to Flux..."
    flux batch submit_test_flux.sh
    
else
    echo "❌ No job scheduler detected!"
    echo "💡 Please submit manually:"
    echo "   - For Tioga: flux batch submit_test_tioga.sh"
    echo "   - For Slurm: sbatch submit_test_slurm.sh"
    echo "   - For Flux: flux batch submit_test_flux.sh"
    exit 1
fi

echo "✅ Job submitted successfully!"
echo "📊 Monitor with:"
echo "   - Flux: flux jobs -u all"
echo "   - Slurm: squeue -u $USER"
echo "📝 Check logs in: logs/ directory"
