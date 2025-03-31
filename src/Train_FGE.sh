#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=yolo_fge_aug
#SBATCH --output=/cluster/home/ishfaqab/Saithes_prepared/logsFGE/FGE_%j.log
#SBATCH --error=/cluster/home/ishfaqab/Saithes_prepared/logsFGE/FGE_%j.err
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=60:00:00
#SBATCH --mem=32G

# Create log directories if not existing
mkdir -p /cluster/home/ishfaqab/Saithes_prepared/logsFGE

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/home/ishfaqab/miniconda3/envs/saithe_env

# Export NVIDIA specific configurations
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Change to project directory
cd /cluster/home/ishfaqab/Saithes_prepared

# Print system information
echo "=== Environment Information ==="
nvidia-smi
echo "==========================="
python /cluster/home/ishfaqab/Saithes_prepared/dataIfiltered/Train_SWA_N.py \
    --data_path /cluster/home/ishfaqab/Saithes_prepared/dataIfiltered/data.yaml \
    --model_path yolov8m.pt \
    --save_dir /cluster/home/ishfaqab/Saithes_prepared/fge_checkpoints \
    --epochs 300 \
    --base_epoch 260 \
    --batch_size 32 \
    --image_size 640 \
    --base_lr 0.0001 \
    --max_lr 0.01

echo "Training completed. Check logs for details."