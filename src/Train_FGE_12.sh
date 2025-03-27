#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=saithe_fge
#SBATCH --output=/cluster/home/ishfaqab/Saithes_prepared/logsFGE/FGE_%j.log
#SBATCH --error=/cluster/home/ishfaqab/Saithes_prepared/logsFGE/FGE_%j.err
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=60:00:00

# Create log directories
mkdir -p /cluster/home/ishfaqab/FGE_EPOCH_12nano

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/home/ishfaqab/miniconda3/envs/saithe_env

# Ensure you're in the right directory
cd /cluster/home/ishfaqab/Saithes_prepared

# Print system info
echo "=== Environment Information ==="
nvidia-smi
echo "==========================="

# Run training with output logging
python /cluster/home/ishfaqab/Saithes_prepared/dataIfiltered/Train_FGE_12.py \
    --data_path /cluster/home/ishfaqab/Saithes_prepared/dataIfiltered/data.yaml \
    --model_path yolo12m.pt \
    --save_dir /cluster/home/ishfaqab/FGE_EPOCH_12m_340\
    --epochs 340 \
    --base_epoch 300 \
    --cycle 4 \
    --batch_size 32 \
    --image_size 640 \
    --base_lr 0.0001 \
    --max_lr 0.01 \
    --device 0 \
    --workers 4 \
    --augment 