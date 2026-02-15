#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C "geforce3090|a5000|a5500"
#SBATCH -t 2:00:00
#SBATCH --mem=48G
#SBATCH -J eval-gen
#SBATCH -o slurm-eval-%j.out

source ~/pytorch.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u eval_generator.py
