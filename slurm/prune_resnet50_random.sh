#!/bin/bash
#SBATCH --job-name=prune_resnet50_random
#SBATCH --account=bewo-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/prune_resnet50_random_%j.out
#SBATCH --error=logs/slurm/prune_resnet50_random_%j.err

module load cuda

CONDA_PYTHON="/u/ssoma1/.conda/envs/torchprune/bin/python"

$CONDA_PYTHON scripts/prune_resnet50.py \
    --data-path /work/hdd/bewo/mahdi/imagenet \
    --pruning-method random \
    --pruning-ratio 0.5 \
    --batch-size 64 \
    --save-dir output/pruned
