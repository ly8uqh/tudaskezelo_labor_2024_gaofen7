#!/bin/bash

#SBATCH --job-name="test2"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=120G
#SBATCH --account=c_dl_brats
#SBATCH --time=01:00:00
#SBATCH --output=/project/c_dl_brats/banmarton/test2/slurm_log/stdout-%x_%j.log
#SBATCH --error=/project/c_dl_brats/banmarton/test2/slurm_log/stderr-%x_%j.log

export WANDB_API_KEY=1b3379bb1f53c415083377d6e778ad51d63f0c77

srun python3 /home/c_dl_bm/banmarton/test2/src/train.py
