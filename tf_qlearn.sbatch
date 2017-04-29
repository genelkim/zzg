#!/bin/sh
#SBATCH -J tf_qlearn
#SBATCH -o my_output%j
#SBATCH -e my_error%j
#SBATCH --gres=gpu:1
#SBATCH -c 2
module load python/2.7.12 cuda/8.0 cudnn/8.0
python tf_project.py
