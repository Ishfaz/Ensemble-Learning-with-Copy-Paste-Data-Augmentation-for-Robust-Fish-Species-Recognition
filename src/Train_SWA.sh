#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=swa_yolo
#SBATCH --output=/cluster/home/ishfaqab/Saithes_prepared/logsSWA/SWA_%j.log
#SBATCH --error=/cluster/home/ishfaqab/Saithes_prepared/logsSWA/SWA_%j.err
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00  # Doubled time for SWA
#SBATCH --mem=64G        # Increased memory for SWA model copies

# Create log directories if they don't exist
mkdir -p /cluster/home/ishfaqab/Saithes_prepared/logsSWA

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/home/ishfaqab/miniconda3/envs/saithe_env

# NVIDIA and PyTorch configurations
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1

# Change to the project directory
cd /cluster/home/ishfaqab/Saithes_prepared

# Print system information
echo "=== Environment Information ==="
nvidia-smi
echo "==========================="

# Run the training script with SWA parameters
python /cluster/home/ishfaqab/Saithes_prepared/dataIfiltered/Train_SWA.py\
