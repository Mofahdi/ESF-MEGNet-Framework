#!/bin/bash
#SBATCH --job-name=megnet
#SBATCH --time=10:30:00
#SBATCH --output=results.out
#SBATCH --error=results.err
#SBATCH --nodes=1
##SBATCH --gres=gpu:2g.20gb:1
#SBATCH --gpus=1


python main.py

