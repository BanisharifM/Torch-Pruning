#!/bin/bash
#SBATCH --job-name=eval_resnet50_l2
#SBATCH --account=bewo-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/eval_resnet50_l2_%j.out
#SBATCH --error=logs/slurm/eval_resnet50_l2_%j.err

module load cuda

CONDA_PYTHON="/u/ssoma1/.conda/envs/torchprune/bin/python"

$CONDA_PYTHON scripts/eval_resnet50.py \
    --data-path /work/hdd/bewo/mahdi/imagenet \
    --ckpt output/pruned/resnet50_l2.pth \
    --batch-size 64
