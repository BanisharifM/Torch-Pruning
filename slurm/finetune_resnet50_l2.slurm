#!/bin/bash
#SBATCH --job-name=finetune_resnet50_l2
#SBATCH --account=bewo-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/finetune_resnet50_l2_%j.out
#SBATCH --error=logs/slurm/finetune_resnet50_l2_%j.err

module load cuda
CONDA_PYTHON="/u/ssoma1/.conda/envs/torchprune/bin/python"

export NCCL_DEBUG=INFO

$CONDA_PYTHON -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_port=23400 \
  finetune_pruned_resnet.py \
  --model-path output/pruned/resnet50_l2.pth \
  --data-path /work/hdd/bewo/mahdi/imagenet \
  --output-dir output/finetuned/resnet50_l2 \
  --epochs 300 \
  --batch-size 256 \
  --lr 0.0005 \
  --weight-decay 0.05
