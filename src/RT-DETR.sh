#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=saithe_detection
#SBATCH --output=/cluster/home/ishfaqab/Saithes_prepared/logsRT/DTR_%j.log
#SBATCH --error=/cluster/home/ishfaqab/Saithes_prepared/logsRT/_DTR_%j.err
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=60:00:00


# Load necessary module
# # Activate conda environment
# source ~/cluster/home/ishfaqab/miniconda3/envs/saithe_env

# source /cluster/home/ishfaqab/miniconda3/etc/profile.d/conda.sh
# conda init bash
conda activate /cluster/home/ishfaqab/miniconda3/envs/saithe_env

# Ensure you're in the right directory
cd /cluster/home/ishfaqab/Saithes_prepared/

# Run the Python script
python /cluster/home/ishfaqab/Saithes_prepared/dataIfiltered/RT-DETR.py