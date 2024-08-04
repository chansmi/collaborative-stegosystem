#!/bin/bash

# Activate the conda environment
source /usr/workspace/$USER/x86_miniconda/bin/activate collaborative-stegosystem

# Set up environment variables
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Print some debug information
echo "Number of nodes: $FLUX_JOB_NUM_NODES"
echo "Total number of tasks: $FLUX_JOB_SIZE"
echo "GPUs per node: 8"
echo "Total number of GPUs: $(($FLUX_JOB_NUM_NODES * 8))"

# Run the distributed training script
flux run -n $FLUX_JOB_SIZE python main.py --config config.yaml

# Deactivate the conda environment
conda deactivate