#!/bin/bash
#SBATCH --job-name=finetune_resnet50_l1
#SBATCH --account=bewo-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/finetune_resnet50_l1_%j.out
#SBATCH --error=logs/slurm/finetune_resnet50_l1_%j.err

module load cuda
CONDA_PYTHON="/u/ssoma1/.conda/envs/torchprune/bin/python"

# export NCCL_DEBUG=INFO

$CONDA_PYTHON scripts/finetune_pruned_resnet.py \
  --model-path output/pruned/resnet50_l1.pth \
  --data-path /work/hdd/bewo/mahdi/imagenet \
  --output-dir output/finetuned/resnet50_l1 \
  --epochs 4 \
  --batch-size 256 \
  --lr 0.0005 \
  --weight-decay 0.05
